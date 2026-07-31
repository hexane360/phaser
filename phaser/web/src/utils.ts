
// App root (any `root_path`/SCRIPT_NAME mount prefix, no trailing slash), recovered from
// the current page's path. `/job/<id>` dashboards strip those two segments; every other
// page is served from the root itself.
export function rootPrefix(): string {
    const parts = window.location.pathname.replace(/\/$/, '').split('/');
    if (parts.length >= 3 && parts[parts.length - 2] === 'job') {
        parts.pop();
        parts.pop();
    }
    return parts.join('/');
}

export type ArrayOrNum = number | ReadonlyArray<number>;

export function isClose<T extends ArrayOrNum>(left: T, right: T, rtol: number = 1e-6, atol: number = 1e-6): boolean {
    if (typeof left == "number") {
        return typeof right == "number" && (
            Math.abs(left - right) < Math.max(rtol * Math.max(Math.abs(left), Math.abs(right)), atol)
        );
    }
    if (typeof right == "number" || left.length != right.length) return false;
    for (let i = 0; i < left.length; i++) {
        if (!isClose(left[i], right[i], rtol, atol)) return false;
    }
    return true;
}