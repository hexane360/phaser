import React from 'react';
import { atom } from 'jotai';

import { ActionIcon, AppShell, Splitter, Tooltip } from '@mantine/core';
import { useLocalStorage, useMediaQuery } from '@mantine/hooks';
import { IconLayoutSidebar } from '@tabler/icons-react';

import Header from '../header';
import { usePubSubConnection } from '../pubsub';
import { ConnectionStatus } from '../connection';
import { Layout, addWidget, defaultLayout, mergePanes, parseLayout, setPaneWidths, splitWith } from './layout';
import { DragPayload, dragEnd, subscribeDrag } from './dnd';
import { LayoutProvider } from './context';
import { Pane } from './Pane';
import { Palette } from './Palette';
import { JobCancelButton, JobErrorBanner, JobStatusText } from './JobStatus';
import classes from './Layout.module.css';

// Right-edge hot zone, live only while something is being dragged. The preview is a
// separate, non-interactive element so it can be as wide as the pane the drop will create
// (`share`) without the hot zone itself having to be.
function SplitZone({share, onDrop}: {share: number, onDrop: () => void}) {
    const [active, setActive] = React.useState(false);

    return <>
        <div
            className={classes.splitZone}
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setActive(true); }}
            onDragLeave={() => setActive(false)}
            onDrop={(e) => { e.preventDefault(); setActive(false); onDrop(); dragEnd(); }}
        />
        {active && <div className={classes.splitPreview} style={{width: `${share}%`}}/>}
    </>;
}

export function Dashboard(props: {}) {
    const conn = usePubSubConnection();
    const [fallbackStatus] = React.useState(() => atom<ConnectionStatus>({ type: 'connecting' }));

    const desktop = useMediaQuery('(min-width: 48em)', true, {getInitialValueInEffect: false});
    // computed once: the default is only a fallback, and re-deriving it on every resize
    // would fight the stored layout
    const [fallback] = React.useState(() => defaultLayout(!!desktop));

    // One global key, not per-job: users want the same working layout on every job.
    const [layout, setLayout] = useLocalStorage<Layout>({
        key: 'phaser.dashboard.layout',
        defaultValue: fallback,
        getInitialValueInEffect: false,
        deserialize: (raw) => parseLayout(raw ? JSON.parse(raw) : undefined, fallback),
    });
    const [paletteOpen, setPaletteOpen] = useLocalStorage<boolean>({
        key: 'phaser.dashboard.palette',
        defaultValue: true,
        getInitialValueInEffect: false,
    });
    const [focusedPane, focusPane] = React.useState(0);

    const update = React.useCallback(
        (fn: (layout: Layout) => Layout) => setLayout((prev) => fn(prev)),
        [setLayout],
    );
    const api = React.useMemo(
        () => ({layout, update, focusedPane: Math.min(focusedPane, layout.panes.length - 1), focusPane}),
        [layout, update, focusedPane],
    );

    // palette clicks append to the focused pane's active tab
    const addToFocused = (viewKey: string) => update((layout) => {
        const pane = layout.panes[Math.min(focusedPane, layout.panes.length - 1)];
        const tab = pane.tabs[pane.active];
        return tab ? addWidget(layout, tab.id, tab.widgets.length, viewKey) : layout;
    });

    const split = layout.panes.length > 1;

    // Live pane widths during a drag. `Splitter` fires `onSizeChange` on every pointer move,
    // which would mean a localStorage write per frame, so those land here and only
    // `onResizeEnd` commits them to `layout`. The length check drops a stale override left
    // by a pane being added or dropped mid-drag.
    const [dragSizes, setDragSizes] = React.useState<Array<number> | null>(null);
    const dragging = React.useRef(false);
    const widths = layout.panes.map((pane) => pane.width);
    const sizes = dragSizes?.length === widths.length ? dragSizes : widths;

    // anything dropped against the right edge opens a new pane holding it. The zone
    // exists only when `splitWith` would actually change something -- an unchanged layout
    // is how every mutation here reports "refused", so the preview and the drop can't
    // disagree (e.g. a pane's only tab, which can't move out without emptying it).
    const [drag, setDrag] = React.useState<DragPayload | null>(null);
    React.useEffect(() => subscribeDrag(setDrag), []);
    const splitTarget = desktop && drag ? splitWith(layout, drag) : layout;
    const canSplitByDrag = splitTarget !== layout;

    const splitByDrag = () => { if (drag) update((layout) => splitWith(layout, drag)); };

    // a phone is too narrow for side-by-side panes; merge them rather than hiding any, so
    // no tab becomes unreachable
    React.useEffect(() => {
        if (!desktop && split) update(mergePanes);
    }, [desktop, split, update]);

    const actions = <>
        <Tooltip label="Toggle view palette">
            <ActionIcon size="xl" visibleFrom="sm" aria-label="Toggle view palette" onClick={() => setPaletteOpen((open) => !open)}>
                <IconLayoutSidebar size="80%"/>
            </ActionIcon>
        </Tooltip>
    </>;

    return <AppShell header={{ height: 80 }} padding={0}>
        <AppShell.Header>
            <Header
                serverStatus={conn?.status ?? fallbackStatus} info={<><JobStatusText/> <JobCancelButton/></>}
                actions={actions} size="100%"
            />
        </AppShell.Header>
        <AppShell.Main className={classes.main}>
            <LayoutProvider value={api}>
                <JobErrorBanner/>
                <div className={classes.workspace}>
                    {paletteOpen && <Palette onPick={addToFocused}/>}
                    <div className={classes.panes}>
                        {split
                            ? <Splitter
                                className={classes.splitter} sizes={sizes}
                                onResizeStart={() => { dragging.current = true; }}
                                onSizeChange={(next) => {
                                    // keyboard and double-click-reset resizes fire only this,
                                    // never `onResizeEnd`, and are one event per gesture --
                                    // so persist them straight away
                                    if (dragging.current) setDragSizes(next as Array<number>);
                                    else update((layout) => setPaneWidths(layout, next as Array<number>));
                                }}
                                onResizeEnd={(_, next) => {
                                    dragging.current = false;
                                    setDragSizes(null);
                                    update((layout) => setPaneWidths(layout, next as Array<number>));
                                }}
                            >
                                {layout.panes.map((pane, paneIdx) =>
                                    <Splitter.Pane key={pane.id} className={classes.splitterPane} defaultSize={pane.width} min={10}>
                                        <Pane paneIdx={paneIdx} pane={pane}/>
                                    </Splitter.Pane>
                                )}
                            </Splitter>
                            : <Pane paneIdx={0} pane={layout.panes[0]}/>}
                        {canSplitByDrag && <SplitZone share={100 / (layout.panes.length + 1)} onDrop={splitByDrag}/>}
                    </div>
                </div>
            </LayoutProvider>
        </AppShell.Main>
    </AppShell>;
}
