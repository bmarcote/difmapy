// The glass background of difmapy's windows on macOS, loaded with ctypes by
// difmapy.plots.macglass (no PyObjC needed). Built into
// python/difmapy/_glass.dylib by build_native.py and the app build.
//
// difmapy_glass(view, kind) puts an effect view behind the window's Qt
// content - a Liquid Glass NSGlassEffectView where AppKit has one (macOS 26
// and later), the translucent window material (NSVisualEffectView)
// otherwise - and makes the window see-through, so that whatever the Qt
// widgets leave unpainted shows the glass. It returns 2 for glass, 1 for the
// window material, 0 if there was no window to work on.
#import <AppKit/AppKit.h>

static NSString *const kTag = @"difmapy.glass";

int difmapy_glass(void *viewptr, int kind) {
    if (viewptr == NULL) {
        return 0;
    }
    __block int result = 0;
    void (^work)(void) = ^{
        NSView *view = (__bridge NSView *)viewptr;
        NSWindow *window = view.window;
        if (window == nil) {
            return;
        }
        NSView *content = window.contentView;
        NSView *frame = content.superview;
        if (frame == nil) {
            return;
        }
        for (NSView *sub in frame.subviews) {
            if ([sub.identifier isEqualToString:kTag]) {
                result = [sub isKindOfClass:[NSVisualEffectView class]] ? 1 : 2;
                return;  // already done
            }
        }
        NSView *effect = nil;
        Class glass = NSClassFromString(@"NSGlassEffectView");
        if (glass != nil && kind != 1) {
            effect = [[glass alloc] initWithFrame:frame.bounds];
            result = 2;
        } else {
            NSVisualEffectView *v = [[NSVisualEffectView alloc] initWithFrame:frame.bounds];
            v.material = NSVisualEffectMaterialUnderWindowBackground;
            v.blendingMode = NSVisualEffectBlendingModeBehindWindow;
            v.state = NSVisualEffectStateFollowsWindowActiveState;
            effect = v;
            result = 1;
        }
        effect.identifier = kTag;
        effect.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [frame addSubview:effect positioned:NSWindowBelow relativeTo:content];
        window.opaque = NO;
        window.backgroundColor = NSColor.clearColor;
        window.titlebarAppearsTransparent = YES;
    };
    if ([NSThread isMainThread]) {
        work();
    } else {
        dispatch_sync(dispatch_get_main_queue(), work);
    }
    return result;
}
