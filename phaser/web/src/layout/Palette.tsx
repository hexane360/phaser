import React from 'react';

import { VIEWS } from '../views';
import { ViewChip } from './ViewPicker';
import classes from './Layout.module.css';
import { Title } from '@mantine/core';

export function Palette({onPick}: {onPick: (viewKey: string) => void}) {
    return <div className={classes.palette}>
        <Title order={3}>Views</Title>
        {Array.from(VIEWS.values(), (view) => <ViewChip key={view.key} view={view} onPick={() => onPick(view.key)}/>)}
    </div>;
}
