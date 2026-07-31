import React from 'react';

import { Title, Text } from '@mantine/core';

import { VIEWS } from '../views';
import { View } from '../views/types';
import { dragStart, dragEnd } from './dnd';
import classes from './Layout.module.css';

// A draggable/clickable palette entry. Shared by `Palette` and `ViewPicker` so the two can
// never drift apart.
export function ViewChip({view, onPick}: {view: View, onPick: () => void}) {
    return <button
        className={classes.chip} onClick={onPick} draggable
        onDragStart={(e) => dragStart(e, {kind: 'view', view: view.key}, view.name)}
        onDragEnd={dragEnd}
    >
        <span className={classes.chipName}>{view.name}</span>
        <span className={classes.chipDesc}>{view.description}</span>
    </button>;
}

// Body of an empty tab: the view list itself, rather than a modal (the only way to add a
// view on mobile, where the palette is hidden).
export function ViewPicker({onPick}: {onPick: (viewKey: string) => void}) {
    return <div className={classes.newtabPage}>
        <Title order={3}>Add a view</Title>
        <Text size="sm" c="dimmed">Pick a view to add it to this tab, or drag one in from the palette.</Text>
        <div className={classes.pickerGrid}>
            {Array.from(VIEWS.values(), (view) => <ViewChip key={view.key} view={view} onPick={() => onPick(view.key)}/>)}
        </div>
    </div>;
}
