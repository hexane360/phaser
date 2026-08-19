import React from 'react';

import { useLayout } from './context';
import { Tab, addWidget, moveWidget } from './layout';
import { dragEnd, dragPayload, insertionIndexY } from './dnd';
import { Widget } from './Widget';
import { ViewPicker } from './ViewPicker';
import classes from './Layout.module.css';

export function Panel({tab}: {tab: Tab}) {
    const {update} = useLayout();
    const ref = React.useRef<HTMLDivElement | null>(null);
    const [dropIndex, setDropIndex] = React.useState<number | null>(null);

    const handleDragOver = (event: React.DragEvent) => {
        const payload = dragPayload();
        if (!payload || payload.kind === 'tab') return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
        setDropIndex(insertionIndexY(ref.current!.querySelectorAll<HTMLElement>('[data-widget]'), event.clientY));
    };

    const handleDrop = (event: React.DragEvent) => {
        const payload = dragPayload();
        const index = dropIndex ?? tab.widgets.length;
        setDropIndex(null);
        if (!payload || payload.kind === 'tab') return;
        event.preventDefault();

        if (payload.kind === 'view') {
            update((layout) => addWidget(layout, tab.id, index, payload.view));
        } else {
            update((layout) => moveWidget(layout, {tabId: payload.tabId, index: payload.index}, {tabId: tab.id, index}));
        }
        dragEnd();
    };

    const dropline = <div className={classes.dropline}/>;

    return <div
        ref={ref} className={classes.panel}
        onDragOver={handleDragOver}
        onDragLeave={(e) => { if (!ref.current!.contains(e.relatedTarget as Node)) setDropIndex(null); }}
        onDrop={handleDrop}
    >
        {tab.widgets.length === 0
            ? <ViewPicker onPick={(viewKey) => update((layout) => addWidget(layout, tab.id, 0, viewKey))}/>
            : tab.widgets.map((widget, index) => <React.Fragment key={widget.id}>
                {dropIndex === index && dropline}
                <Widget tabId={tab.id} widget={widget} index={index}/>
            </React.Fragment>)}
        {dropIndex !== null && dropIndex >= tab.widgets.length && dropline}
    </div>;
}
