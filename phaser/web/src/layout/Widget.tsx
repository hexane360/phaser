import React from 'react';

import { Title } from '@mantine/core';
import { IconChevronDown, IconGripVertical, IconX } from '@tabler/icons-react';

import { VIEWS } from '../views';
import { ViewParams } from '../views/types';
import { useLayout } from './context';
import { Widget as WidgetState, removeWidget, setWidgetParams, toggleWidget } from './layout';
import { dragEnd, dragStart } from './dnd';
import classes from './Layout.module.css';

interface WidgetProps {
    tabId: string;
    widget: WidgetState;
    index: number;
}

export function Widget({tabId, widget, index}: WidgetProps) {
    const {update} = useLayout();

    const setParams = React.useCallback(
        (params: ViewParams) => update((layout) => setWidgetParams(layout, tabId, widget.id, params)),
        [update, tabId, widget.id],
    );

    const view = VIEWS.get(widget.view);
    if (!view) return null;

    return <div className={classes.widget} data-widget data-open={widget.open}>
        <div
            className={classes.widgetHead} draggable
            onClick={() => update((layout) => toggleWidget(layout, tabId, widget.id))}
            onDragStart={(e) => dragStart(e, {kind: 'widget', tabId, index}, view.name)}
            onDragEnd={dragEnd}
        >
            <span className={classes.grip}><IconGripVertical size={18}/></span>
            <span className={classes.chev}><IconChevronDown size={20}/></span>
            <Title order={3} className={classes.widgetTitle}>{view.name}</Title>
            <span className={classes.spacer}/>
            <span
                className={classes.widgetClose} role="button" aria-label={`Close ${view.name}`}
                onClick={(e) => { e.stopPropagation(); update((layout) => removeWidget(layout, tabId, widget.id)); }}
            ><IconX size={22}/></span>
        </div>
        <div className={classes.widgetBody}>
            <view.Component params={widget.params} setParams={setParams}/>
        </div>
    </div>;
}
