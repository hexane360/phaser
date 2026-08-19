import { View } from './types';
import { ProgressView } from './progress';
import { ObjectAmpMeanView, ObjectAmpStackView, ObjectPhaseSumView, ObjectPhaseStackView } from './object';
import { ProbeModesView, ProbeModesRecipView, ProbeSumView, ProbeSumRecipView } from './probe';
import { PositionsView } from './positions';
import { LogsView } from './logs';

export type { View, ViewProps, ViewParams } from './types';

// Registry keys are persisted in the stored layout; renaming one orphans existing widgets
// (`parseLayout` drops them).
export const VIEWS: Map<string, View> = new Map((<Array<View>>[
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
        key: 'objectAmpMean',
        name: 'Object amplitude mean',
        description: 'Object amplitude, geometric mean over slices',
        Component: ObjectAmpMeanView,
        topicLabel: () => 'obj_amp_mean',
    },
    {
        key: 'objectAmpStack',
        name: 'Object amplitude stack',
        description: 'Object amplitude stack',
        Component: ObjectAmpStackView,
        defaultParams: {slice: 0},
        topicLabel: (params) => `obj?slice=${params.slice ?? 0}`,
    },
    {
        key: 'probes',
        name: 'Probe modes',
        description: 'Phase/amplitude plot of each probe mode',
        Component: ProbeModesView,
        defaultParams: {mode: 'phaseAmp'},
        topicLabel: () => 'probes',
    },
    {
        key: 'probesRecip',
        name: 'Probe modes (recip.)',
        description: 'Phase/amplitude plot of each probe mode in reciprocal space',
        Component: ProbeModesRecipView,
        defaultParams: {mode: 'phaseAmp'},
        topicLabel: () => 'probes_recip',
    },
    {
        key: 'probeSum',
        name: 'Total probe intensity',
        description: 'Total probe intensity in real space',
        Component: ProbeSumView,
        topicLabel: () => 'probe_sum',
    },
    {
        key: 'probeSumRecip',
        name: 'Total probe intensity (recip.)',
        description: 'Total probe intensity in reciprocal space',
        Component: ProbeSumRecipView,
        topicLabel: () => 'probe_sum_recip',
    },
    {
        key: 'positions',
        name: 'Probe positions',
        description: 'Scan positions, in scan order',
        Component: PositionsView,
        topicLabel: () => 'positions',
    },
    {
        key: 'logs',
        name: 'Logs',
        description: 'Live reconstruction log',
        Component: LogsView,
        topicLabel: () => 'logs',
    },
]).map((view) => [view.key, view]));
