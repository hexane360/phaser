// Transient failure reporting. Durable state -- a job that failed, and is still failed after
// a reload -- belongs in a banner instead (`layout/JobStatus.tsx`).

import { notifications } from '@mantine/notifications';

const AUTO_CLOSE_MS = 8_000;

// Shows a failure as a red notification. `message` comes from `errorMessage` (`requests.tsx`).
export function reportError(title: string, message: string): void {
    console.error(`${title}: ${message}`);
    notifications.show({color: 'red', title, message, autoClose: AUTO_CLOSE_MS});
}
