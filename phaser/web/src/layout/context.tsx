import React from 'react';

import { Layout } from './layout';

export interface LayoutApi {
    layout: Layout;
    // applies a pure mutation from `layout.ts`
    update: (fn: (layout: Layout) => Layout) => void;
    // pane a palette click adds to; the last pane the user interacted with
    focusedPane: number;
    focusPane: (paneIdx: number) => void;
}

const LayoutContext = React.createContext<LayoutApi | null>(null);

export function LayoutProvider({value, children}: React.PropsWithChildren<{value: LayoutApi}>) {
    return <LayoutContext.Provider value={value}>{children}</LayoutContext.Provider>;
}

export function useLayout(): LayoutApi {
    const api = React.useContext(LayoutContext);
    if (!api) throw new Error("useLayout() outside a LayoutProvider");
    return api;
}
