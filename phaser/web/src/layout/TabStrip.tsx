import React from 'react';

import { IconX } from '@tabler/icons-react';

import { VIEWS } from '../views';
import { useLayout } from './context';
import {
    Pane, Tab, addTab, addWidget, closeTab, makeTab, makeWidget, moveTab, moveWidget,
    setActiveTab, tabLabel, widgetToNewTab,
} from './layout';
import { dragEnd, dragPayload, dragStart, insertionIndexX } from './dnd';
import classes from './Layout.module.css';

export function TabStrip({paneIdx, pane}: {paneIdx: number, pane: Pane}) {
    const {update} = useLayout();
    const ref = React.useRef<HTMLDivElement | null>(null);
    const [dropIndex, setDropIndex] = React.useState<number | null>(null);

    // Drops that reach the strip itself: a tab being reordered/moved between panes, or a
    // view/widget dropped past the end of the tab list (-> a new tab holding it).
    const handleDragOver = (event: React.DragEvent) => {
        const payload = dragPayload();
        if (!payload) return;

        // A view/widget over a tab belongs to that tab, which highlights itself and has
        // already called `preventDefault`. `TabButton` lets the event bubble here anyway so
        // the end-of-list marker gets cleared -- `dragleave` can't do it, since moving onto
        // a tab never leaves the strip.
        if (payload.kind !== 'tab' && (event.target as Element).closest?.('[data-tab]')) {
            setDropIndex(null);
            return;
        }

        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
        setDropIndex(payload.kind === 'tab'
            ? insertionIndexX(ref.current!.querySelectorAll<HTMLElement>('[data-tab]'), event.clientX)
            : pane.tabs.length);
    };

    const handleDrop = (event: React.DragEvent) => {
        const payload = dragPayload();
        const tabIdx = dropIndex ?? pane.tabs.length;
        setDropIndex(null);
        if (!payload) return;
        event.preventDefault();

        if (payload.kind === 'tab') {
            update((layout) => moveTab(layout, {paneIdx: payload.paneIdx, tabIdx: payload.tabIdx}, {paneIdx, tabIdx}));
        } else if (payload.kind === 'widget') {
            update((layout) => widgetToNewTab(layout, {tabId: payload.tabId, index: payload.index}, paneIdx));
        } else {
            const view = VIEWS.get(payload.view);
            if (view) update((layout) => addTab(layout, paneIdx, makeTab([makeWidget(view.key)])));
        }
        dragEnd();
    };

    // zero-width, so showing it can't shift the tabs the insertion index is measured from
    const marker = <div className={classes.tabDropline}/>;

    return <div
        ref={ref} className={classes.tabstrip}
        onDragOver={handleDragOver}
        onDragLeave={(e) => { if (!ref.current!.contains(e.relatedTarget as Node)) setDropIndex(null); }}
        onDrop={handleDrop}
    >
        {pane.tabs.map((tab, tabIdx) => <React.Fragment key={tab.id}>
            {dropIndex === tabIdx && marker}
            <TabButton tab={tab} paneIdx={paneIdx} tabIdx={tabIdx} active={tabIdx === pane.active}/>
        </React.Fragment>)}
        {dropIndex !== null && dropIndex >= pane.tabs.length && marker}
        <button className={classes.newtab} aria-label="New tab" onClick={() => update((layout) => addTab(layout, paneIdx))}>+</button>
    </div>;
}

interface TabButtonProps {
    tab: Tab;
    paneIdx: number;
    tabIdx: number;
    active: boolean;
}

function TabButton({tab, paneIdx, tabIdx, active}: TabButtonProps) {
    const {update} = useLayout();
    const [dropping, setDropping] = React.useState(false);
    const label = tabLabel(tab);

    // views and widgets dropped on a tab append to it; a dropped *tab* falls through to the
    // strip, which positions it by pointer x
    const handleDragOver = (event: React.DragEvent) => {
        const payload = dragPayload();
        if (!payload || payload.kind === 'tab') return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
        setDropping(true);
    };

    const handleDrop = (event: React.DragEvent) => {
        const payload = dragPayload();
        setDropping(false);
        if (!payload || payload.kind === 'tab') return;
        event.preventDefault();
        event.stopPropagation();

        if (payload.kind === 'view') {
            update((layout) => setActiveTab(addWidget(layout, tab.id, tab.widgets.length, payload.view), paneIdx, tabIdx));
        } else {
            update((layout) => setActiveTab(
                moveWidget(layout, {tabId: payload.tabId, index: payload.index}, {tabId: tab.id, index: tab.widgets.length}),
                paneIdx, tabIdx,
            ));
        }
        dragEnd();
    };

    return <div
        className={`${classes.tab} ${dropping ? classes.dropping : ''}`} data-tab data-active={active} role="tab"
        title={label} draggable
        onClick={() => update((layout) => setActiveTab(layout, paneIdx, tabIdx))}
        onDragStart={(e) => dragStart(e, {kind: 'tab', paneIdx, tabIdx}, label)}
        onDragEnd={dragEnd}
        onDragOver={handleDragOver}
        onDragLeave={() => setDropping(false)}
        onDrop={handleDrop}
    >
        <span className={classes.tabLabel}>{label}</span>
        {tab.widgets.length > 0 && <span className={classes.count}>{tab.widgets.length}</span>}
        <span
            className={classes.tabClose} role="button" aria-label={`Close ${label}`}
            onClick={(e) => { e.stopPropagation(); update((layout) => closeTab(layout, paneIdx, tabIdx)); }}
        ><IconX size={13}/></span>
    </div>;
}
