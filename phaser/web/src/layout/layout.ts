// Persisted dashboard layout, plus pure mutations over it. Every function here is
// `(layout, ...args) => Layout`, so `Dashboard` can drive all of them through one
// `setLayout`.

import { VIEWS } from '../views';
import { ViewParams } from '../views/types';
import type { DragPayload } from './dnd';

export const LAYOUT_VERSION = 1;

export interface Widget {
    id: string;
    view: string;
    params: ViewParams;
    open: boolean;
}

export interface Tab {
    id: string;
    widgets: Array<Widget>;
}

export interface Pane {
    id: string;
    tabs: Array<Tab>;
    active: number;
    width: number;  // percentage of the workspace; `normalizeWidths` keeps these summing to 100
}

export interface Layout {
    version: typeof LAYOUT_VERSION;
    panes: Array<Pane>;  // at least one
}

export interface WidgetAddr { tabId: string; index: number; }
export interface TabAddr { paneIdx: number; tabIdx: number; }

let idCounter = 0;

// Widget/tab ids are React keys *and* are persisted, so a fresh session's ids must not
// collide with ids restored from localStorage -- hence the timestamp, not a bare counter.
export function newId(): string {
    return `${Date.now().toString(36)}-${(idCounter++).toString(36)}`;
}

export function makeWidget(viewKey: string, params?: ViewParams): Widget {
    return {
        id: newId(),
        view: viewKey,
        params: params ?? {...(VIEWS.get(viewKey)?.defaultParams ?? {})},
        open: true,
    };
}

export function makeTab(widgets: Array<Widget> = []): Tab {
    return { id: newId(), widgets };
}

export function makePane(tabs: Array<Tab>, width: number): Pane {
    return { id: newId(), tabs, active: 0, width };
}

// Width for a pane about to be appended: `100 / n` against `n` panes already summing to
// 100 normalizes to `100 / (n + 1)` each, so every pane ends up with an equal share.
export function newPaneWidth(layout: Layout): number {
    return 100. / layout.panes.length;
}

export const NEW_TAB_TITLE = 'New tab';

// Tabs aren't named, they're described by what they hold -- so the label tracks widgets
// being added, removed, and reordered with no state of its own.
export function tabLabel(tab: Tab): string {
    if (!tab.widgets.length) return NEW_TAB_TITLE;
    return tab.widgets.map((widget) => VIEWS.get(widget.view)?.name ?? widget.view).join(', ');
}

export function defaultLayout(desktop: boolean): Layout {
    if (desktop) {
        const widgets = ['progress', 'object', 'probe', 'logs'].map((key) => makeWidget(key));
        return {
            version: LAYOUT_VERSION,
            panes: [makePane([makeTab(widgets)], 100.)],
         };
    }
    // one widget per tab: a phone can't usefully scroll a four-plot accordion
    const tabs = ['progress', 'object', 'probe', 'logs'].map((key) =>
        makeTab([makeWidget(key)])
    );
    return {
        version: LAYOUT_VERSION,
        panes: [makePane(tabs, 100.)],
    };
}

// Drops widgets whose view no longer exists, clamps `active`, and falls back to
// `fallback` on anything structurally wrong (including a version bump).
export function parseLayout(raw: unknown, fallback: Layout): Layout {
    try {
        const layout = raw as Layout;
        if (!layout || layout.version !== LAYOUT_VERSION || !Array.isArray(layout.panes)) return fallback;

        const panes = layout.panes.map((pane): Pane => {
            const tabs = (Array.isArray(pane.tabs) ? pane.tabs : []).map((tab): Tab => ({
                id: typeof tab.id === 'string' ? tab.id : newId(),
                widgets: (Array.isArray(tab.widgets) ? tab.widgets : [])
                    .filter((widget) => widget && VIEWS.has(widget.view))
                    .map((widget): Widget => ({
                        id: typeof widget.id === 'string' ? widget.id : newId(),
                        view: widget.view,
                        params: (widget.params && typeof widget.params === 'object') ? widget.params : {},
                        open: widget.open !== false,
                    })),
            }));
            // a stored width can be anything; an unusable one falls back to an equal share
            const width = Number.isFinite(pane.width) && pane.width > 0. ? pane.width : 1.;
            const id = typeof pane.id === 'string' ? pane.id : newId();
            return { id, tabs, active: clamp(pane.active ?? 0, tabs.length), width };
        }).filter((pane) => pane.tabs.length > 0);

        return panes.length ? { version: LAYOUT_VERSION, panes: normalizeWidths(panes) } : fallback;
    } catch {
        return fallback;
    }
}

function clamp(index: number, length: number): number {
    return Math.min(Math.max(Math.trunc(index) || 0, 0), Math.max(length - 1, 0));
}

// Widths only mean anything relative to each other, so anything that adds or drops a pane
// renormalizes to keep them summing to 100 (leaving a lone pane at 100). Also the guard
// against a nonsense total -- NaN or 0 would otherwise poison every width.
function normalizeWidths(panes: Array<Pane>): Array<Pane> {
    const total = panes.reduce((sum, pane) => sum + pane.width, 0.);
    if (!(total > 0.)) return panes.map((pane) => ({ ...pane, width: 100. / panes.length }));
    return panes.map((pane) => ({ ...pane, width: pane.width * 100. / total }));
}

function mapPane(layout: Layout, paneIdx: number, fn: (pane: Pane) => Pane): Layout {
    if (!layout.panes[paneIdx]) return layout;
    return { ...layout, panes: layout.panes.map((pane, i) => i === paneIdx ? fn(pane) : pane) };
}

function mapTab(layout: Layout, tabId: string, fn: (tab: Tab) => Tab): Layout {
    return {
        ...layout,
        panes: layout.panes.map((pane) => ({
            ...pane,
            tabs: pane.tabs.map((tab) => tab.id === tabId ? fn(tab) : tab),
        })),
    };
}

export function findTab(layout: Layout, tabId: string): Tab | undefined {
    for (const pane of layout.panes) {
        const tab = pane.tabs.find((tab) => tab.id === tabId);
        if (tab) return tab;
    }
    return undefined;
}

function insertAt<T>(items: ReadonlyArray<T>, index: number, item: T): Array<T> {
    const at = Math.min(Math.max(index, 0), items.length);
    return [...items.slice(0, at), item, ...items.slice(at)];
}

export function addWidget(layout: Layout, tabId: string, index: number, viewKey: string, params?: ViewParams): Layout {
    if (!VIEWS.has(viewKey)) return layout;
    const widget = makeWidget(viewKey, params);
    return mapTab(layout, tabId, (tab) => ({
        ...tab,
        widgets: insertAt(tab.widgets, index, widget),
    }));
}

export function removeWidget(layout: Layout, tabId: string, widgetId: string): Layout {
    return mapTab(layout, tabId, (tab) => ({ ...tab, widgets: tab.widgets.filter((w) => w.id !== widgetId) }));
}

export function moveWidget(layout: Layout, src: WidgetAddr, dst: WidgetAddr): Layout {
    const srcTab = findTab(layout, src.tabId);
    const widget = srcTab?.widgets[src.index];
    if (!widget) return layout;

    // within one tab, removing first shifts every later index down by one
    const dstIndex = src.tabId === dst.tabId && src.index < dst.index ? dst.index - 1 : dst.index;
    const removed = mapTab(layout, src.tabId, (tab) => ({
        ...tab, widgets: tab.widgets.filter((_, i) => i !== src.index),
    }));
    return mapTab(removed, dst.tabId, (tab) => ({ ...tab, widgets: insertAt(tab.widgets, dstIndex, widget) }));
}

// Moves a widget out into a tab of its own, appended to `paneIdx`'s strip. Emptying the
// source tab is fine -- it falls back to showing the view picker.
export function widgetToNewTab(layout: Layout, src: WidgetAddr, paneIdx: number): Layout {
    const widget = findTab(layout, src.tabId)?.widgets[src.index];
    if (!widget) return layout;

    const removed = mapTab(layout, src.tabId, (tab) => ({
        ...tab, widgets: tab.widgets.filter((_, i) => i !== src.index),
    }));
    return addTab(removed, paneIdx, makeTab([widget]));
}

export function toggleWidget(layout: Layout, tabId: string, widgetId: string): Layout {
    return mapTab(layout, tabId, (tab) => ({
        ...tab,
        widgets: tab.widgets.map((w) => w.id === widgetId ? { ...w, open: !w.open } : w),
    }));
}

export function setWidgetParams(layout: Layout, tabId: string, widgetId: string, params: ViewParams): Layout {
    return mapTab(layout, tabId, (tab) => ({
        ...tab,
        widgets: tab.widgets.map((w) => w.id === widgetId ? { ...w, params } : w),
    }));
}

export function addTab(layout: Layout, paneIdx: number, tab: Tab = makeTab()): Layout {
    return mapPane(layout, paneIdx, (pane) => ({ ...pane, tabs: [...pane.tabs, tab], active: pane.tabs.length }));
}

// Closing a pane's last tab collapses the pane (ending a split); the sole remaining pane
// instead keeps one empty tab, so there's always somewhere to add a view.
export function closeTab(layout: Layout, paneIdx: number, tabIdx: number): Layout {
    const pane = layout.panes[paneIdx];
    if (!pane || !pane.tabs[tabIdx]) return layout;

    const tabs = pane.tabs.filter((_, i) => i !== tabIdx);
    if (!tabs.length && layout.panes.length > 1) {
        return { ...layout, panes: normalizeWidths(layout.panes.filter((_, i) => i !== paneIdx)) };
    }
    const nextTabs = tabs.length ? tabs : [makeTab()];
    return mapPane(layout, paneIdx, (prev) => ({ ...prev, tabs: nextTabs, active: clamp(pane.active > tabIdx ? pane.active - 1 : pane.active, nextTabs.length) }));
}

export function moveTab(layout: Layout, src: TabAddr, dst: TabAddr): Layout {
    const tab = layout.panes[src.paneIdx]?.tabs[src.tabIdx];
    if (!tab) return layout;
    if (src.paneIdx === dst.paneIdx && (src.tabIdx === dst.tabIdx || src.tabIdx === dst.tabIdx - 1)) return layout;

    const dstIdx = src.paneIdx === dst.paneIdx && src.tabIdx < dst.tabIdx ? dst.tabIdx - 1 : dst.tabIdx;
    const removed = mapPane(layout, src.paneIdx, (pane) => {
        const tabs = pane.tabs.filter((_, i) => i !== src.tabIdx);
        return { ...pane, tabs, active: clamp(pane.active > src.tabIdx ? pane.active - 1 : pane.active, tabs.length) };
    });
    // the source pane may have just emptied; drop it and shift the destination index down
    const emptied = removed.panes[src.paneIdx].tabs.length === 0 && removed.panes.length > 1;
    const compacted = emptied ? { ...removed, panes: normalizeWidths(removed.panes.filter((_, i) => i !== src.paneIdx)) } : removed;
    const dstPaneIdx = emptied && dst.paneIdx > src.paneIdx ? dst.paneIdx - 1 : dst.paneIdx;

    return mapPane(compacted, dstPaneIdx, (pane) => ({
        ...pane,
        tabs: insertAt(pane.tabs, dstIdx, tab),
        active: Math.min(dstIdx, pane.tabs.length),
    }));
}

export function setActiveTab(layout: Layout, paneIdx: number, tabIdx: number): Layout {
    return mapPane(layout, paneIdx, (pane) => ({ ...pane, active: clamp(tabIdx, pane.tabs.length) }));
}

// Collapses every pane into the first, for a viewport too narrow to hold more than one.
export function mergePanes(layout: Layout): Layout {
    if (layout.panes.length < 2) return layout;
    const [first, ...rest] = layout.panes;
    const tabs = [...first.tabs, ...rest.flatMap((pane) => pane.tabs)];
    return { ...layout, panes: [{ ...first, tabs, width: 100. }] };
}

// Commits dragged pane widths. Ignores a size list that doesn't match the current panes:
// a resize can land after a split/merge has already changed their number.
export function setPaneWidths(layout: Layout, widths: ReadonlyArray<number>): Layout {
    if (widths.length !== layout.panes.length) return layout;
    const panes = layout.panes.map((pane, i): Pane => ({ ...pane, width: widths[i] }));
    return { ...layout, panes: normalizeWidths(panes) };
}

// Right-edge drop: moves the dragged tab into a pane of its own. Refused when it would
// empty the source pane, which would just be dropped again.
export function splitWithTab(layout: Layout, src: TabAddr): Layout {
    const pane = layout.panes[src.paneIdx];
    const tab = pane?.tabs[src.tabIdx];
    if (!tab || pane.tabs.length < 2) return layout;

    const width = newPaneWidth(layout);
    const tabs = pane.tabs.filter((_, i) => i !== src.tabIdx);
    const kept = mapPane(layout, src.paneIdx, (prev) => ({
        ...prev, tabs, active: clamp(prev.active > src.tabIdx ? prev.active - 1 : prev.active, tabs.length),
    }));
    return { ...kept, panes: normalizeWidths([...kept.panes, makePane([tab], width)]) };
}

// Right-edge drop: moves the dragged widget into a pane of its own.
export function splitWithWidget(layout: Layout, src: WidgetAddr): Layout {
    const widget = findTab(layout, src.tabId)?.widgets[src.index];
    if (!widget) return layout;

    const width = newPaneWidth(layout);
    const removed = mapTab(layout, src.tabId, (tab) => ({
        ...tab, widgets: tab.widgets.filter((_, i) => i !== src.index),
    }));
    return { ...removed, panes: normalizeWidths([...removed.panes, makePane([makeTab([widget])], width)]) };
}

// Right-edge drop for anything draggable. Returns `layout` unchanged when the drop would
// do nothing -- which is how the drop zone decides whether to offer a preview at all, so
// the preview can never promise a split that the drop then refuses.
export function splitWith(layout: Layout, payload: DragPayload): Layout {
    switch (payload.kind) {
        case 'view': {
            const view = VIEWS.get(payload.view);
            if (!view) return layout;
            const width = newPaneWidth(layout);
            const tab = makeTab([makeWidget(view.key)]);
            return { ...layout, panes: normalizeWidths([...layout.panes, makePane([tab], width)]) };
        }
        case 'widget':
            return splitWithWidget(layout, {tabId: payload.tabId, index: payload.index});
        case 'tab':
            return splitWithTab(layout, {paneIdx: payload.paneIdx, tabIdx: payload.tabIdx});
    }
}
