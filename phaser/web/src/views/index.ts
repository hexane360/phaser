import { View } from './types';
import { ProgressView } from './progress';
import { ObjectPhaseSumView } from './objectPhaseSum';
import { ObjectPhaseStackView } from './objectPhaseStack';
import { ProbeModesView } from './probe';
import { LogsView } from './logs';

export type { View, ViewProps, ViewParams } from './types';

// Registry keys are persisted in the stored layout; renaming one orphans existing widgets
// (`parseLayout` drops them).
export const VIEWS: Map<string, View> = new Map(([
    {
        key: 'progress',
        name: 'Progress',
        description: 'Total loss per iteration',
        Component: ProgressView,
        topicLabel: () => 'progress',
    },
    {
        key: 'objectPhaseSum',
        name: 'Object phase sum',
        description: 'Object phase, summed over slices',
        Component: ObjectPhaseSumView,
        topicLabel: () => 'obj_phase_sum',
    },
    {
        key: 'objectSlice',
        name: 'Object phase stack',
        description: 'Object phase stack',
        Component: ObjectPhaseStackView,
        defaultParams: {slice: 0},
        topicLabel: (params) => `obj?slice=${params.slice ?? 0}`,
    },
    {
        key: 'probes',
        name: 'Probe modes',
        description: 'Complex plot of each probe mode',
        Component: ProbeModesView,
        topicLabel: () => 'probes',
    },
    {
        key: 'logs',
        name: 'Logs',
        description: 'Live reconstruction log',
        Component: LogsView,
        topicLabel: () => 'logs',
    },
] satisfies Array<View>).map((view) => [view.key, view]));
