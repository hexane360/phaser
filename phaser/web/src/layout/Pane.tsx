import React from 'react';

import { useLayout } from './context';
import { Pane as PaneState } from './layout';
import { TabStrip } from './TabStrip';
import { Panel } from './Panel';
import classes from './Layout.module.css';

export function Pane({paneIdx, pane}: {paneIdx: number, pane: PaneState}) {
    const {focusPane} = useLayout();
    const tab = pane.tabs[pane.active];

    return <div className={classes.pane} onPointerDown={() => focusPane(paneIdx)}>
        <TabStrip paneIdx={paneIdx} pane={pane}/>
        {tab && <Panel key={tab.id} tab={tab}/>}
    </div>;
}
