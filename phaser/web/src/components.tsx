import { Collapse, Title } from '@mantine/core';
import React from 'react';
import { useDisclosure } from '@mantine/hooks';

export interface BoxProps {
    crossAlign?: 'center' | 'start' | 'end' | 'stretch'; // = 'center'

    children?: React.ReactNode;
}

export const HBox = React.forwardRef<HTMLDivElement, BoxProps>(function HBox(props, ref) {
    const style = {
        flexDirection: 'row',
        alignItems: props.crossAlign ?? 'center',
    } as const;

    return <div className="hbox" style={style} ref={ref}>
        { props.children }
    </div>;
});

export const VBox = React.forwardRef<HTMLDivElement, BoxProps>(function (props, ref) {
    const style = {
        flexDirection: 'column',
        alignItems: props.crossAlign ?? 'center',
    } as const;

    return <div className="vbox" style={style} ref={ref}>
        { props.children }
    </div>;
});

interface SectionProps {
    name: string
    children?: React.ReactNode
}

export function Section(props: SectionProps) {
    const [opened, {toggle}] = useDisclosure(true);

    return <>
        <div className="section-header" onClick={toggle}><Title order={3}>{ props.name }</Title></div>
        <Collapse className="section" expanded={opened}>{ props.children }</Collapse>
    </>;
}

interface MonoProps {
    children?: React.ReactNode;
}

export function Mono({children}: MonoProps) {
    return <span className="mono">{children}</span>;
}