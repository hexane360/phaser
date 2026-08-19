// Plain HTML5 drag and drop for the dashboard: palette chips into panels/tabs, widgets
// between tabs, tabs between panes.

import React from 'react';

export type DragPayload =
    | { kind: 'view'; view: string }                      // from the palette or picker
    | { kind: 'widget'; tabId: string; index: number }
    | { kind: 'tab'; paneIdx: number; tabIdx: number };

// Module-level rather than React state: `dragover` handlers must read this synchronously,
// and a drag starting must not re-render the tree. `dataTransfer` can't stand in -- its
// data is unreadable during `dragover` in every browser.
let payload: DragPayload | null = null;

// Drop targets that only exist *during* a drag (the right-edge split zone) need to
// re-render when one starts, which the module-level `payload` alone can't trigger.
// `dragstart` can't be watched on `window` instead: `dragStart` stops propagation, so the
// native event never gets past React's root container.
type DragListener = (payload: DragPayload | null) => void;
const listeners = new Set<DragListener>();

export function dragStart(event: React.DragEvent, value: DragPayload, label: string) {
    event.stopPropagation();
    payload = value;
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', label);
    for (const listener of listeners) listener(payload);
}

export function dragEnd() {
    payload = null;
    for (const listener of listeners) listener(null);
}

export function dragPayload(): DragPayload | null {
    return payload;
}

export function subscribeDrag(listener: DragListener): () => void {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
}

// Index at which a drop at `clientY` should insert, comparing against each child's vertical
// midpoint.
export function insertionIndexY(children: ArrayLike<HTMLElement>, clientY: number): number {
    for (let i = 0; i < children.length; i++) {
        const rect = children[i].getBoundingClientRect();
        if (clientY < rect.top + rect.height / 2) return i;
    }
    return children.length;
}

// Horizontal equivalent, for tab strips.
export function insertionIndexX(children: ArrayLike<HTMLElement>, clientX: number): number {
    for (let i = 0; i < children.length; i++) {
        const rect = children[i].getBoundingClientRect();
        if (clientX < rect.left + rect.width / 2) return i;
    }
    return children.length;
}
