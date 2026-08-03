
export function rootPrefix(): string {
    const meta = document.querySelector<HTMLMetaElement>('meta[name="phaser-root"]');
    if (!meta) {
        console.error("Missing `phaser-root` meta tag");
        return "";
    }
    return meta.content.replace(/\/$/, '');
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

// Time since the job started, always unit-labelled: a `H:MM:SS` reading would be taken
// for a wall clock, which is the confusion this column exists to avoid.
export function formatElapsed(seconds: number): string {
    const round1 = (x: number) => Math.round(x * 10.) / 10.;

    // each unit is chosen from the value already rounded to the digit shown, so 3599.98 s
    // reads "1 h 0.0 min" rather than "60.0 min"
    if (round1(seconds) < 60.) return `${round1(seconds).toFixed(1)} s`;
    const minutes = round1(seconds / 60.);
    if (minutes < 60.) return `${minutes.toFixed(1)} min`;
    const hours = round1(minutes / 60.);
    if (hours < 24.) {
        const whole = Math.floor(minutes / 60.);
        return `${whole} h ${(minutes - 60. * whole).toFixed(1)} min`;
    }
    const days = Math.floor(hours / 24.);
    return `${days} d ${(hours - 24. * days).toFixed(1)} h`;
}

// `HH:MM:SS.mmm` -- at 10 records/s a second's resolution isn't enough to order them
export function formatTime(timestamp: string): string {
    const date = new Date(timestamp);
    const [h, m, s] = [date.getHours(), date.getMinutes(), date.getSeconds()].map((n) => String(n).padStart(2, '0'));
    return `${h}:${m}:${s}.${String(date.getMilliseconds()).padStart(3, '0')}`;
}