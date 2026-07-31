import React from 'react';

import { TopicKey } from '../types';

// A widget's persisted configuration, e.g. `{slice: 0}`.
export type ViewParams = Record<string, TopicKey>;

export interface ViewProps {
    params: ViewParams;
    setParams: (params: ViewParams) => void;
}

export interface View {
    // registry key; also what the persisted layout stores
    key: string;
    // short label: palette entry, tab title, widget header
    name: string;
    // one line, shown under the name in the palette and picker
    description: string;
    // rendered inside the widget body
    Component: React.ComponentType<ViewProps>;
    // starting params for a freshly-added widget
    defaultParams?: ViewParams;
    // cosmetic subscription label for the widget header, e.g. `obj?slice=0`
    topicLabel?: (params: ViewParams) => string;
}
