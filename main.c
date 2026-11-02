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
 */
#define _GNU_SOURCE
#include <errno.h>
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

#include "color-management-v1-client-protocol.h"
#include "xdg-decoration-unstable-v1-client-protocol.h"
#include "xdg-shell-client-protocol.h"

#define TEXT        "This text is bright red"
#define DEFAULT_NITS 400.0f
#define SCRGB_NITS_PER_UNIT 80.0f

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

static void render(struct buffer *b)
{
	int w = b->width, h = b->height;

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
	 * scales light directly. Background is 0.0 (black), alpha 1.0. */
	const float peak = target_nits / SCRGB_NITS_PER_UNIT; /* 400 nits = 5.0 */
	uint16_t lut[256];
	for (int i = 0; i < 256; i++)
		lut[i] = float_to_half(peak * (i / 255.0f));
	const uint16_t zero = float_to_half(0.0f);
	const uint16_t one = float_to_half(1.0f);

	/* DRM ABGR16161616F, little-endian: memory order R, G, B, A. */
	uint16_t *dst = b->data;
	for (int y = 0; y < h; y++) {
		const uint8_t *s = src + (size_t)y * src_stride;
		uint16_t *d = dst + (size_t)y * w * 4;
		for (int x = 0; x < w; x++) {
			d[x * 4 + 0] = lut[s[x]];
			d[x * 4 + 1] = zero;
			d[x * 4 + 2] = zero;
			d[x * 4 + 3] = one;
		}
	}

	cairo_surface_destroy(mask);
}

static void redraw(void)
{
	if (!app.current || app.current->width != app.width ||
	    app.current->height != app.height || app.current->busy) {
		buffer_retire(app.current);
		app.current = buffer_create(app.width, app.height);
		if (!app.current) {
			app.running = false;
			return;
		}
	}
	render(app.current);
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

int main(int argc, char **argv)
{
	if (argc > 1) {
		char *end;
		target_nits = strtof(argv[1], &end);
		if (*end || !(target_nits >= 0.0f)) {
			fprintf(stderr, "usage: %s [nits]  (default %.0f)\n", argv[0], DEFAULT_NITS);
			return 1;
		}
	}

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
	if (!app.shm_has_fp16) {
		fprintf(stderr, "wl_shm lacks ABGR16161616F (FP16) format\n");
		return 1;
	}

	app.surface = wl_compositor_create_surface(app.compositor);

	/* Color management: Windows-scRGB image description on the surface. */
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

	/* Resizable toplevel. */
	app.xdg_surface = xdg_wm_base_get_xdg_surface(app.wm_base, app.surface);
	xdg_surface_add_listener(app.xdg_surface, &xdg_surface_listener, NULL);
	app.toplevel = xdg_surface_get_toplevel(app.xdg_surface);
	xdg_toplevel_add_listener(app.toplevel, &toplevel_listener, NULL);
	char title[64];
	snprintf(title, sizeof(title), "scRGB test (%g nit red)", target_nits);
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
