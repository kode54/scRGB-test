/*
 * scrgb-test: minimal Wayland Windows-scRGB test client.
 *
 * Creates a resizable xdg_toplevel, tags its wl_surface with the
 * wp_color_manager_v1.create_windows_scrgb image description, and fills an
 * FP16 (ABGR16161616F) wl_shm buffer with a black frame containing the text
 * "This text is bright red" at 400 cd/m² (or the nits given as the first argument).
 *
 * Windows-scRGB: BT.709 primaries, extended linear, 1.0 == 80 cd/m².
 * So 400 cd/m² red == (5.0, 0.0, 0.0).
 *
 * --vulkan presents through a real Vulkan swapchain and makes the same choices
 * as DXGI (DXVK, vkd3d-proton) and Wine's winewayland.drv:
 *
 *   - If the driver lists R16G16B16A16_SFLOAT + EXTENDED_SRGB_LINEAR, use
 *     PASS_THROUGH and tag the surface with create_windows_scrgb.
 *   - Otherwise DXGI silently falls back to G22, which gives an FP16
 *     SRGB_NONLINEAR swapchain. The application adds no image description,
 *     and the NVIDIA driver tags the surface as parametric sRGB by itself.
 *
 * Mesa only lists EXTENDED_SRGB_LINEAR when the compositor advertises the
 * extended_target_volume feature, and the NVIDIA driver behaves the same way.
 * Without it, the fallback gets linear values decoded as if they were sRGB-encoded:
 * values below 1.0 turn darker and values above 1.0 much brighter.
 *
 * --assume-scrgb takes the tagged path even when the format is not listed. It
 * sends the same pixels through the same swapchain code, so it is the control
 * for --vulkan.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cairo.h>
#include <wayland-client.h>
#define VK_USE_PLATFORM_WAYLAND_KHR
#include <vulkan/vulkan.h>

#include "color-management-v1-client-protocol.h"
#include "xdg-decoration-unstable-v1-client-protocol.h"
#include "xdg-shell-client-protocol.h"

#define TEXT        "This text is bright red"
#define DEFAULT_NITS 400.0f
#define SCRGB_NITS_PER_UNIT 80.0f
#define DEFAULT_VULKAN_BACKGROUND_NITS 24.0f
#define MAX_SWAPCHAIN_IMAGES 16

struct buffer {
	struct wl_buffer *wl_buffer;
	void *data;
	size_t size;
	int width, height;
	bool busy;
	bool stale;
};

static struct {
	struct wl_display *display;
	struct wl_registry *registry;
	struct wl_compositor *compositor;
	struct wl_shm *shm;
	struct xdg_wm_base *wm_base;
	struct wp_color_manager_v1 *color_manager;
	struct zxdg_decoration_manager_v1 *decoration_manager;

	bool shm_has_fp16;
	bool cm_has_scrgb;
	bool cm_has_extended_target_volume;
	bool cm_has_perceptual;
	bool cm_done;

	struct wl_surface *surface;
	struct xdg_surface *xdg_surface;
	struct xdg_toplevel *toplevel;
	struct wp_color_management_surface_v1 *cm_surface;
	struct wp_image_description_v1 *image_desc;
	int desc_state; /* 0 pending, 1 ready, -1 failed */

	int pending_w, pending_h;
	int width, height;
	bool configured;
	bool running;

	struct buffer *current;
} app = { .running = true };

static float target_nits = DEFAULT_NITS;
static float background_nits = -1.0f; /* negative: use the mode's default */
static bool use_vulkan;
static bool assume_scrgb;

/* ---------- float -> IEEE 754 binary16 ---------- */

static uint16_t float_to_half(float f)
{
	union { float f; uint32_t u; } v = { f };
	uint32_t sign = (v.u >> 16) & 0x8000;
	int32_t exp = (int32_t)((v.u >> 23) & 0xff) - 127 + 15;
	uint32_t mant = v.u & 0x7fffff;

	if (((v.u >> 23) & 0xff) == 0xff) /* inf / nan */
		return sign | 0x7c00 | (mant ? 0x200 : 0);
	if (exp >= 31)
		return sign | 0x7c00;
	if (exp <= 0) {
		if (exp < -10)
			return sign;
		mant |= 0x800000;
		uint32_t shift = 14 - exp;
		uint32_t half = mant >> shift;
		uint32_t rem = mant & ((1u << shift) - 1);
		uint32_t mid = 1u << (shift - 1);
		if (rem > mid || (rem == mid && (half & 1)))
			half++;
		return sign | half;
	}
	uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
	uint32_t rem = mant & 0x1fff;
	if (rem > 0x1000 || (rem == 0x1000 && (half & 1)))
		half++; /* may carry into exponent, which is correct */
	return half;
}

/* ---------- buffers ---------- */

static void buffer_release(void *data, struct wl_buffer *wl_buffer)
{
	(void)wl_buffer;
	struct buffer *b = data;
	b->busy = false;
	if (b->stale) {
		wl_buffer_destroy(b->wl_buffer);
		munmap(b->data, b->size);
		free(b);
	}
}

static const struct wl_buffer_listener buffer_listener = {
	.release = buffer_release,
};

static struct buffer *buffer_create(int width, int height)
{
	int stride = width * 8; /* 4 x half */
	size_t size = (size_t)stride * height;

	int fd = memfd_create("scrgb-test", MFD_CLOEXEC);
	if (fd < 0) {
		perror("memfd_create");
		return NULL;
	}
	if (ftruncate(fd, size) < 0) {
		perror("ftruncate");
		close(fd);
		return NULL;
	}
	void *data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (data == MAP_FAILED) {
		perror("mmap");
		close(fd);
		return NULL;
	}

	struct wl_shm_pool *pool = wl_shm_create_pool(app.shm, fd, size);
	struct buffer *b = calloc(1, sizeof(*b));
	b->wl_buffer = wl_shm_pool_create_buffer(pool, 0, width, height, stride,
	                                         WL_SHM_FORMAT_ABGR16161616F);
	wl_shm_pool_destroy(pool);
	close(fd);

	b->data = data;
	b->size = size;
	b->width = width;
	b->height = height;
	wl_buffer_add_listener(b->wl_buffer, &buffer_listener, b);
	return b;
}

static void buffer_retire(struct buffer *b)
{
	if (!b)
		return;
	if (b->busy) {
		b->stale = true;
		return;
	}
	wl_buffer_destroy(b->wl_buffer);
	munmap(b->data, b->size);
	free(b);
}

/* ---------- rendering ---------- */

static void render(void *pixels, int w, int h)
{
	/* Rasterize anti-aliased text coverage into an A8 mask with cairo. */
	cairo_surface_t *mask = cairo_image_surface_create(CAIRO_FORMAT_A8, w, h);
	cairo_t *cr = cairo_create(mask);
	cairo_select_font_face(cr, "sans-serif", CAIRO_FONT_SLANT_NORMAL,
	                       CAIRO_FONT_WEIGHT_BOLD);

	cairo_text_extents_t ext;
	cairo_set_font_size(cr, 100.0);
	cairo_text_extents(cr, TEXT, &ext);
	double size = 100.0 * (w * 0.85) / ext.width;
	double max_size = h * 0.4;
	if (size > max_size)
		size = max_size;
	cairo_set_font_size(cr, size);
	cairo_text_extents(cr, TEXT, &ext);

	cairo_move_to(cr, (w - ext.width) / 2.0 - ext.x_bearing,
	              (h - ext.height) / 2.0 - ext.y_bearing);
	cairo_set_source_rgba(cr, 0, 0, 0, 1);
	cairo_show_text(cr, TEXT);
	cairo_destroy(cr);
	cairo_surface_flush(mask);

	const uint8_t *src = cairo_image_surface_get_data(mask);
	int src_stride = cairo_image_surface_get_stride(mask);

	/* Precompute all 256 coverage levels: linear scRGB, so coverage
	 * scales light directly. Background is neutral gray, alpha 1.0. */
	const float peak = target_nits / SCRGB_NITS_PER_UNIT; /* 400 nits = 5.0 */
	const float background = background_nits / SCRGB_NITS_PER_UNIT;
	uint16_t lut_red[256], lut_gray[256];
	for (int i = 0; i < 256; i++) {
		float coverage = i / 255.0f;
		lut_gray[i] = float_to_half(background * (1.0f - coverage));
		lut_red[i] = float_to_half(background * (1.0f - coverage) + peak * coverage);
	}
	const uint16_t one = float_to_half(1.0f);

	/* DRM ABGR16161616F and VK R16G16B16A16_SFLOAT: memory order R, G, B, A. */
	uint16_t *dst = pixels;
	for (int y = 0; y < h; y++) {
		const uint8_t *s = src + (size_t)y * src_stride;
		uint16_t *d = dst + (size_t)y * w * 4;
		for (int x = 0; x < w; x++) {
			d[x * 4 + 0] = lut_red[s[x]];
			d[x * 4 + 1] = lut_gray[s[x]];
			d[x * 4 + 2] = lut_gray[s[x]];
			d[x * 4 + 3] = one;
		}
	}

	cairo_surface_destroy(mask);
}

/* ---------- Vulkan presentation ---------- */

static struct {
	VkInstance instance;
	VkSurfaceKHR surface;
	VkPhysicalDevice physical_device;
	uint32_t queue_family;
	VkDevice device;
	VkQueue queue;
	VkCommandPool command_pool;
	VkCommandBuffer command_buffer;
	VkFence fence;
	VkSemaphore acquire_semaphore;
	VkSemaphore present_semaphores[MAX_SWAPCHAIN_IMAGES];
	VkColorSpaceKHR color_space;
	VkSwapchainKHR swapchain;
	VkImage images[MAX_SWAPCHAIN_IMAGES];
	uint32_t image_count;
	int swapchain_width, swapchain_height;
	VkBuffer staging;
	VkDeviceMemory staging_memory;
	void *staging_map;
	VkDeviceSize staging_size;
} vk;

static bool vk_check(VkResult result, const char *what)
{
	if (result == VK_SUCCESS)
		return true;
	fprintf(stderr, "%s failed: VkResult %d\n", what, result);
	return false;
}

/* Reports what the driver offers and returns whether to tag the surface. */
static bool vk_choose_color_space(bool *tag_surface)
{
	uint32_t count = 0;
	if (!vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, vk.surface,
	                                                   &count, NULL),
	              "vkGetPhysicalDeviceSurfaceFormatsKHR"))
		return false;
	VkSurfaceFormatKHR *formats = calloc(count, sizeof(*formats));
	if (!formats)
		return false;
	bool ok = vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, vk.surface,
	                                                        &count, formats),
	                   "vkGetPhysicalDeviceSurfaceFormatsKHR");

	bool has_scrgb = false, has_pass_through = false, has_srgb = false;
	for (uint32_t i = 0; ok && i < count; i++) {
		if (formats[i].format != VK_FORMAT_R16G16B16A16_SFLOAT)
			continue;
		if (formats[i].colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
			has_scrgb = true;
		else if (formats[i].colorSpace == VK_COLOR_SPACE_PASS_THROUGH_EXT)
			has_pass_through = true;
		else if (formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			has_srgb = true;
	}
	free(formats);
	if (!ok)
		return false;

	printf("compositor windows_scrgb feature:          %s\n", app.cm_has_scrgb ? "yes" : "no");
	printf("compositor extended_target_volume feature: %s\n",
	       app.cm_has_extended_target_volume ? "yes" : "no");
	printf("driver lists FP16 + EXTENDED_SRGB_LINEAR:  %s\n", has_scrgb ? "yes" : "no");

	*tag_surface = has_scrgb || assume_scrgb;
	if (*tag_surface) {
		/* Wine retries with SRGB_NONLINEAR if PASS_THROUGH is unavailable. */
		vk.color_space = has_pass_through ? VK_COLOR_SPACE_PASS_THROUGH_EXT
		                                  : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
		if (!has_pass_through && !has_srgb) {
			fprintf(stderr, "driver has no usable FP16 color space for the tagged path\n");
			return false;
		}
		printf("path: %s swapchain tagged with create_windows_scrgb%s\n",
		       has_pass_through ? "PASS_THROUGH" : "SRGB_NONLINEAR",
		       has_scrgb ? "" : " (forced by --assume-scrgb)");
		printf("expected: correct picture; gray %g cd/m^2, text %g cd/m^2\n",
		       background_nits, target_nits);
	} else {
		if (!has_srgb) {
			fprintf(stderr, "driver does not list FP16 + SRGB_NONLINEAR\n");
			return false;
		}
		vk.color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
		printf("path: DXGI fallback to G22, FP16 SRGB_NONLINEAR swapchain, no create_windows_scrgb\n");
		printf("expected: BUG; linear values decoded as sRGB, gray below %g cd/m^2, text far above %g cd/m^2\n",
		       background_nits, target_nits);
	}
	fflush(stdout);
	return true;
}

static bool vk_init(bool *tag_surface)
{
	const char *instance_extensions[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
		VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
		VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
	};
	const VkApplicationInfo app_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "scrgb-test",
		.apiVersion = VK_API_VERSION_1_1,
	};
	const VkInstanceCreateInfo instance_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app_info,
		.enabledExtensionCount = 3,
		.ppEnabledExtensionNames = instance_extensions,
	};
	if (!vk_check(vkCreateInstance(&instance_info, NULL, &vk.instance), "vkCreateInstance"))
		return false;

	const VkWaylandSurfaceCreateInfoKHR surface_info = {
		.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
		.display = app.display,
		.surface = app.surface,
	};
	if (!vk_check(vkCreateWaylandSurfaceKHR(vk.instance, &surface_info, NULL, &vk.surface),
	              "vkCreateWaylandSurfaceKHR"))
		return false;

	VkPhysicalDevice devices[16];
	uint32_t device_count = 16;
	VkResult result = vkEnumeratePhysicalDevices(vk.instance, &device_count, devices);
	if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
		vk_check(result, "vkEnumeratePhysicalDevices");
		return false;
	}
	bool found = false;
	for (uint32_t d = 0; d < device_count && !found; d++) {
		VkQueueFamilyProperties families[32];
		uint32_t family_count = 32;
		vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &family_count, families);
		for (uint32_t f = 0; f < family_count && !found; f++) {
			VkBool32 present = VK_FALSE;
			if (vkGetPhysicalDeviceSurfaceSupportKHR(devices[d], f, vk.surface, &present) != VK_SUCCESS)
				continue;
			if (present && (families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				vk.physical_device = devices[d];
				vk.queue_family = f;
				found = true;
			}
		}
	}
	if (!found) {
		fprintf(stderr, "no Vulkan device can present to this surface\n");
		return false;
	}
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(vk.physical_device, &properties);
	printf("device: %s\n", properties.deviceName);

	if (!vk_choose_color_space(tag_surface))
		return false;

	const float priority = 1.0f;
	const VkDeviceQueueCreateInfo queue_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = vk.queue_family,
		.queueCount = 1,
		.pQueuePriorities = &priority,
	};
	const char *device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	const VkDeviceCreateInfo device_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queue_info,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = device_extensions,
	};
	if (!vk_check(vkCreateDevice(vk.physical_device, &device_info, NULL, &vk.device),
	              "vkCreateDevice"))
		return false;
	vkGetDeviceQueue(vk.device, vk.queue_family, 0, &vk.queue);

	const VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = vk.queue_family,
	};
	if (!vk_check(vkCreateCommandPool(vk.device, &pool_info, NULL, &vk.command_pool),
	              "vkCreateCommandPool"))
		return false;
	const VkCommandBufferAllocateInfo command_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	if (!vk_check(vkAllocateCommandBuffers(vk.device, &command_info, &vk.command_buffer),
	              "vkAllocateCommandBuffers"))
		return false;

	const VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	const VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	if (!vk_check(vkCreateFence(vk.device, &fence_info, NULL, &vk.fence), "vkCreateFence") ||
	    !vk_check(vkCreateSemaphore(vk.device, &semaphore_info, NULL, &vk.acquire_semaphore),
	              "vkCreateSemaphore"))
		return false;
	for (int i = 0; i < MAX_SWAPCHAIN_IMAGES; i++)
		if (!vk_check(vkCreateSemaphore(vk.device, &semaphore_info, NULL, &vk.present_semaphores[i]),
		              "vkCreateSemaphore"))
			return false;
	return true;
}

static bool vk_create_staging(int width, int height)
{
	VkDeviceSize size = (VkDeviceSize)width * height * 8;
	if (size <= vk.staging_size)
		return true;
	if (vk.staging) {
		vkDestroyBuffer(vk.device, vk.staging, NULL);
		vkFreeMemory(vk.device, vk.staging_memory, NULL);
		vk.staging = VK_NULL_HANDLE;
		vk.staging_memory = VK_NULL_HANDLE;
		vk.staging_size = 0;
	}

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	if (!vk_check(vkCreateBuffer(vk.device, &buffer_info, NULL, &vk.staging), "vkCreateBuffer"))
		return false;

	VkMemoryRequirements requirements;
	vkGetBufferMemoryRequirements(vk.device, vk.staging, &requirements);
	VkPhysicalDeviceMemoryProperties memory;
	vkGetPhysicalDeviceMemoryProperties(vk.physical_device, &memory);
	const VkMemoryPropertyFlags wanted =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; i++)
		if ((requirements.memoryTypeBits & (1u << i)) &&
		    (memory.memoryTypes[i].propertyFlags & wanted) == wanted)
			type = i;
	if (type == UINT32_MAX) {
		fprintf(stderr, "no host-visible coherent memory type\n");
		return false;
	}
	const VkMemoryAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size,
		.memoryTypeIndex = type,
	};
	if (!vk_check(vkAllocateMemory(vk.device, &allocate_info, NULL, &vk.staging_memory),
	              "vkAllocateMemory") ||
	    !vk_check(vkBindBufferMemory(vk.device, vk.staging, vk.staging_memory, 0),
	              "vkBindBufferMemory") ||
	    !vk_check(vkMapMemory(vk.device, vk.staging_memory, 0, VK_WHOLE_SIZE, 0, &vk.staging_map),
	              "vkMapMemory"))
		return false;
	vk.staging_size = size;
	return true;
}

static bool vk_create_swapchain(int width, int height)
{
	vkDeviceWaitIdle(vk.device);

	VkSurfaceCapabilitiesKHR caps;
	if (!vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, vk.surface, &caps),
	              "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))
		return false;
	if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
		fprintf(stderr, "swapchain images cannot be transfer destinations\n");
		return false;
	}

	uint32_t image_count = caps.minImageCount + 1;
	if (caps.maxImageCount && image_count > caps.maxImageCount)
		image_count = caps.maxImageCount;
	const VkCompositeAlphaFlagBitsKHR alpha =
		(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
			? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
			: VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;

	VkSwapchainKHR old = vk.swapchain;
	const VkSwapchainCreateInfoKHR swapchain_info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = vk.surface,
		.minImageCount = image_count,
		.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
		.imageColorSpace = vk.color_space,
		.imageExtent = { (uint32_t)width, (uint32_t)height },
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.preTransform = caps.currentTransform,
		.compositeAlpha = alpha,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR,
		.clipped = VK_TRUE,
		.oldSwapchain = old,
	};
	VkResult result = vkCreateSwapchainKHR(vk.device, &swapchain_info, NULL, &vk.swapchain);
	if (old)
		vkDestroySwapchainKHR(vk.device, old, NULL);
	if (result != VK_SUCCESS) {
		vk.swapchain = VK_NULL_HANDLE;
		vk_check(result, "vkCreateSwapchainKHR");
		return false;
	}

	vk.image_count = MAX_SWAPCHAIN_IMAGES;
	result = vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &vk.image_count, vk.images);
	if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
		vk_check(result, "vkGetSwapchainImagesKHR");
		return false;
	}
	vk.swapchain_width = width;
	vk.swapchain_height = height;
	return vk_create_staging(width, height);
}

static void vk_image_barrier(VkImage image, VkImageLayout from, VkImageLayout to,
                             VkAccessFlags src_access, VkAccessFlags dst_access,
                             VkPipelineStageFlags dst_stage)
{
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
	};
	vkCmdPipelineBarrier(vk.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, dst_stage, 0,
	                     0, NULL, 0, NULL, 1, &barrier);
}

static bool vk_present(int width, int height)
{
	if ((!vk.swapchain || vk.swapchain_width != width || vk.swapchain_height != height) &&
	    !vk_create_swapchain(width, height))
		return false;

	uint32_t index;
	VkResult result = vkAcquireNextImageKHR(vk.device, vk.swapchain, UINT64_MAX,
	                                        vk.acquire_semaphore, VK_NULL_HANDLE, &index);
	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		if (!vk_create_swapchain(width, height))
			return false;
		result = vkAcquireNextImageKHR(vk.device, vk.swapchain, UINT64_MAX,
		                               vk.acquire_semaphore, VK_NULL_HANDLE, &index);
	}
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		vk_check(result, "vkAcquireNextImageKHR");
		return false;
	}

	render(vk.staging_map, width, height);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (!vk_check(vkBeginCommandBuffer(vk.command_buffer, &begin_info), "vkBeginCommandBuffer"))
		return false;
	vk_image_barrier(vk.images[index], VK_IMAGE_LAYOUT_UNDEFINED,
	                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                 VK_PIPELINE_STAGE_TRANSFER_BIT);
	const VkBufferImageCopy region = {
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { (uint32_t)width, (uint32_t)height, 1 },
	};
	vkCmdCopyBufferToImage(vk.command_buffer, vk.staging, vk.images[index],
	                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	vk_image_barrier(vk.images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
	                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	if (!vk_check(vkEndCommandBuffer(vk.command_buffer), "vkEndCommandBuffer"))
		return false;

	const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	const VkSubmitInfo submit_info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk.acquire_semaphore,
		.pWaitDstStageMask = &wait_stage,
		.commandBufferCount = 1,
		.pCommandBuffers = &vk.command_buffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &vk.present_semaphores[index],
	};
	if (!vk_check(vkQueueSubmit(vk.queue, 1, &submit_info, vk.fence), "vkQueueSubmit"))
		return false;

	const VkPresentInfoKHR present_info = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk.present_semaphores[index],
		.swapchainCount = 1,
		.pSwapchains = &vk.swapchain,
		.pImageIndices = &index,
	};
	result = vkQueuePresentKHR(vk.queue, &present_info);
	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR && result != VK_ERROR_OUT_OF_DATE_KHR) {
		vk_check(result, "vkQueuePresentKHR");
		return false;
	}

	/* Frames are drawn only on resize, so the staging buffer and command
	 * buffer can be reused as soon as this submission finishes. */
	if (!vk_check(vkWaitForFences(vk.device, 1, &vk.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences") ||
	    !vk_check(vkResetFences(vk.device, 1, &vk.fence), "vkResetFences") ||
	    !vk_check(vkResetCommandBuffer(vk.command_buffer, 0), "vkResetCommandBuffer"))
		return false;
	return true;
}

static void redraw(void)
{
	if (use_vulkan) {
		if (!vk_present(app.width, app.height))
			app.running = false;
		return;
	}

	if (!app.current || app.current->width != app.width ||
	    app.current->height != app.height || app.current->busy) {
		buffer_retire(app.current);
		app.current = buffer_create(app.width, app.height);
		if (!app.current) {
			app.running = false;
			return;
		}
	}
	render(app.current->data, app.current->width, app.current->height);
	wl_surface_attach(app.surface, app.current->wl_buffer, 0, 0);
	wl_surface_damage_buffer(app.surface, 0, 0, INT32_MAX, INT32_MAX);
	app.current->busy = true;
	wl_surface_commit(app.surface);
}

/* ---------- xdg-shell ---------- */

static void wm_base_ping(void *data, struct xdg_wm_base *wm, uint32_t serial)
{
	(void)data;
	xdg_wm_base_pong(wm, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
	.ping = wm_base_ping,
};

static void xdg_surface_configure(void *data, struct xdg_surface *xs,
                                  uint32_t serial)
{
	(void)data;
	xdg_surface_ack_configure(xs, serial);

	int w = app.pending_w > 0 ? app.pending_w : 960;
	int h = app.pending_h > 0 ? app.pending_h : 540;
	bool changed = !app.configured || w != app.width || h != app.height;
	app.width = w;
	app.height = h;
	app.configured = true;

	if (changed)
		redraw();
	else
		wl_surface_commit(app.surface);
}

static const struct xdg_surface_listener xdg_surface_listener = {
	.configure = xdg_surface_configure,
};

static void toplevel_configure(void *data, struct xdg_toplevel *tl,
                               int32_t w, int32_t h, struct wl_array *states)
{
	(void)data; (void)tl; (void)states;
	app.pending_w = w;
	app.pending_h = h;
}

static void toplevel_close(void *data, struct xdg_toplevel *tl)
{
	(void)data; (void)tl;
	app.running = false;
}

static void toplevel_configure_bounds(void *data, struct xdg_toplevel *tl,
                                      int32_t w, int32_t h)
{
	(void)data; (void)tl; (void)w; (void)h;
}

static void toplevel_wm_capabilities(void *data, struct xdg_toplevel *tl,
                                     struct wl_array *caps)
{
	(void)data; (void)tl; (void)caps;
}

static const struct xdg_toplevel_listener toplevel_listener = {
	.configure = toplevel_configure,
	.close = toplevel_close,
	.configure_bounds = toplevel_configure_bounds,
	.wm_capabilities = toplevel_wm_capabilities,
};

/* ---------- wl_shm ---------- */

static void shm_format(void *data, struct wl_shm *shm, uint32_t format)
{
	(void)data; (void)shm;
	if (format == WL_SHM_FORMAT_ABGR16161616F)
		app.shm_has_fp16 = true;
}

static const struct wl_shm_listener shm_listener = {
	.format = shm_format,
};

/* ---------- color management ---------- */

static void cm_supported_intent(void *data, struct wp_color_manager_v1 *cm,
                                uint32_t intent)
{
	(void)data; (void)cm;
	if (intent == WP_COLOR_MANAGER_V1_RENDER_INTENT_PERCEPTUAL)
		app.cm_has_perceptual = true;
}

static void cm_supported_feature(void *data, struct wp_color_manager_v1 *cm,
                                 uint32_t feature)
{
	(void)data; (void)cm;
	if (feature == WP_COLOR_MANAGER_V1_FEATURE_WINDOWS_SCRGB)
		app.cm_has_scrgb = true;
	else if (feature == WP_COLOR_MANAGER_V1_FEATURE_EXTENDED_TARGET_VOLUME)
		app.cm_has_extended_target_volume = true;
}

static void cm_supported_tf_named(void *data, struct wp_color_manager_v1 *cm,
                                  uint32_t tf)
{
	(void)data; (void)cm; (void)tf;
}

static void cm_supported_primaries_named(void *data,
                                         struct wp_color_manager_v1 *cm,
                                         uint32_t primaries)
{
	(void)data; (void)cm; (void)primaries;
}

static void cm_done(void *data, struct wp_color_manager_v1 *cm)
{
	(void)data; (void)cm;
	app.cm_done = true;
}

static const struct wp_color_manager_v1_listener cm_listener = {
	.supported_intent = cm_supported_intent,
	.supported_feature = cm_supported_feature,
	.supported_tf_named = cm_supported_tf_named,
	.supported_primaries_named = cm_supported_primaries_named,
	.done = cm_done,
};

static void desc_failed(void *data, struct wp_image_description_v1 *d,
                        uint32_t cause, const char *msg)
{
	(void)data; (void)d;
	fprintf(stderr, "windows_scrgb image description failed (cause %u): %s\n",
	        cause, msg);
	app.desc_state = -1;
}

static void desc_ready(void *data, struct wp_image_description_v1 *d,
                       uint32_t identity)
{
	(void)data; (void)d; (void)identity;
	app.desc_state = 1;
}

static const struct wp_image_description_v1_listener desc_listener = {
	.failed = desc_failed,
	.ready = desc_ready,
};

/* ---------- registry ---------- */

static void registry_global(void *data, struct wl_registry *reg, uint32_t name,
                            const char *iface, uint32_t version)
{
	(void)data;
	if (!strcmp(iface, wl_compositor_interface.name)) {
		app.compositor = wl_registry_bind(reg, name, &wl_compositor_interface,
		                                  version < 4 ? version : 4);
	} else if (!strcmp(iface, wl_shm_interface.name)) {
		app.shm = wl_registry_bind(reg, name, &wl_shm_interface, 1);
		wl_shm_add_listener(app.shm, &shm_listener, NULL);
	} else if (!strcmp(iface, xdg_wm_base_interface.name)) {
		app.wm_base = wl_registry_bind(reg, name, &xdg_wm_base_interface,
		                               version < 5 ? version : 5);
		xdg_wm_base_add_listener(app.wm_base, &wm_base_listener, NULL);
	} else if (!strcmp(iface, wp_color_manager_v1_interface.name)) {
		app.color_manager = wl_registry_bind(reg, name,
		                                     &wp_color_manager_v1_interface, 1);
		wp_color_manager_v1_add_listener(app.color_manager, &cm_listener, NULL);
	} else if (!strcmp(iface, zxdg_decoration_manager_v1_interface.name)) {
		app.decoration_manager = wl_registry_bind(
			reg, name, &zxdg_decoration_manager_v1_interface, 1);
	}
}

static void registry_global_remove(void *data, struct wl_registry *reg,
                                   uint32_t name)
{
	(void)data; (void)reg; (void)name;
}

static const struct wl_registry_listener registry_listener = {
	.global = registry_global,
	.global_remove = registry_global_remove,
};

/* ---------- main ---------- */

static void usage(const char *name)
{
	fprintf(stderr,
	        "usage: %s [--vulkan [--assume-scrgb]] [--background NITS] [NITS]\n"
	        "  NITS               text luminance (default %.0f)\n"
	        "  --background NITS  gray background luminance\n"
	        "                     (default 0, or %.0f with --vulkan)\n"
	        "  --vulkan           present through Vulkan, choosing the color space as\n"
	        "                     DXGI and Wine do from the driver's format list\n"
	        "  --assume-scrgb     with --vulkan, take the Windows-scRGB path even if\n"
	        "                     the driver does not list it (the control case)\n",
	        name, DEFAULT_NITS, DEFAULT_VULKAN_BACKGROUND_NITS);
}

static bool parse_nits(const char *text, float *out)
{
	char *end;
	float value = strtof(text, &end);
	if (end == text || *end || !(value >= 0.0f))
		return false;
	*out = value;
	return true;
}

int main(int argc, char **argv)
{
	static const struct option options[] = {
		{ "vulkan", no_argument, NULL, 'v' },
		{ "assume-scrgb", no_argument, NULL, 'a' },
		{ "background", required_argument, NULL, 'b' },
		{ "help", no_argument, NULL, 'h' },
		{ 0 },
	};
	int option;
	while ((option = getopt_long(argc, argv, "", options, NULL)) != -1) {
		switch (option) {
		case 'v':
			use_vulkan = true;
			break;
		case 'a':
			assume_scrgb = true;
			break;
		case 'b':
			if (!parse_nits(optarg, &background_nits)) {
				usage(argv[0]);
				return 1;
			}
			break;
		case 'h':
			usage(argv[0]);
			return 0;
		default:
			usage(argv[0]);
			return 1;
		}
	}
	if (argc - optind > 1 || (assume_scrgb && !use_vulkan) ||
	    (optind < argc && !parse_nits(argv[optind], &target_nits))) {
		usage(argv[0]);
		return 1;
	}
	if (background_nits < 0.0f)
		background_nits = use_vulkan ? DEFAULT_VULKAN_BACKGROUND_NITS : 0.0f;

	app.display = wl_display_connect(NULL);
	if (!app.display) {
		fprintf(stderr, "cannot connect to Wayland display\n");
		return 1;
	}
	app.registry = wl_display_get_registry(app.display);
	wl_registry_add_listener(app.registry, &registry_listener, NULL);
	wl_display_roundtrip(app.display); /* globals */
	wl_display_roundtrip(app.display); /* shm formats, cm capabilities */

	if (!app.compositor || !app.shm || !app.wm_base) {
		fprintf(stderr, "missing wl_compositor / wl_shm / xdg_wm_base\n");
		return 1;
	}
	if (!app.color_manager) {
		fprintf(stderr, "compositor lacks wp_color_manager_v1\n");
		return 1;
	}
	if (!app.cm_has_scrgb) {
		fprintf(stderr, "wp_color_manager_v1 lacks the windows_scrgb feature\n");
		return 1;
	}
	if (!app.cm_has_perceptual) {
		fprintf(stderr, "wp_color_manager_v1 lacks perceptual render intent\n");
		return 1;
	}
	if (!use_vulkan && !app.shm_has_fp16) {
		fprintf(stderr, "wl_shm lacks ABGR16161616F (FP16) format\n");
		return 1;
	}

	app.surface = wl_compositor_create_surface(app.compositor);

	bool tag_surface = true;
	if (use_vulkan && !vk_init(&tag_surface))
		return 1;

	/* Color management: Windows-scRGB image description on the surface. */
	if (tag_surface) {
		app.cm_surface = wp_color_manager_v1_get_surface(app.color_manager,
		                                                 app.surface);
		app.image_desc = wp_color_manager_v1_create_windows_scrgb(app.color_manager);
		wp_image_description_v1_add_listener(app.image_desc, &desc_listener, NULL);
		while (app.desc_state == 0)
			if (wl_display_dispatch(app.display) < 0) {
				fprintf(stderr, "display error waiting for image description\n");
				return 1;
			}
		if (app.desc_state < 0)
			return 1;
		wp_color_management_surface_v1_set_image_description(
			app.cm_surface, app.image_desc,
			WP_COLOR_MANAGER_V1_RENDER_INTENT_PERCEPTUAL);
	}

	/* Resizable toplevel. */
	app.xdg_surface = xdg_wm_base_get_xdg_surface(app.wm_base, app.surface);
	xdg_surface_add_listener(app.xdg_surface, &xdg_surface_listener, NULL);
	app.toplevel = xdg_surface_get_toplevel(app.xdg_surface);
	xdg_toplevel_add_listener(app.toplevel, &toplevel_listener, NULL);
	char title[96];
	snprintf(title, sizeof(title), "scRGB test (%g nit red, %s)", target_nits,
	         !use_vulkan ? "wl_shm" : tag_surface ? "Vulkan, scRGB" : "Vulkan, sRGB fallback");
	xdg_toplevel_set_title(app.toplevel, title);
	xdg_toplevel_set_app_id(app.toplevel, "scrgb-test");
	xdg_toplevel_set_min_size(app.toplevel, 160, 90);

	if (app.decoration_manager) {
		struct zxdg_toplevel_decoration_v1 *deco =
			zxdg_decoration_manager_v1_get_toplevel_decoration(
				app.decoration_manager, app.toplevel);
		zxdg_toplevel_decoration_v1_set_mode(
			deco, ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
	}

	wl_surface_commit(app.surface); /* initial commit, no buffer */

	while (app.running && wl_display_dispatch(app.display) != -1)
		;

	return 0;
}
