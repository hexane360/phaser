import React from 'react';
import { useAtom } from 'jotai';

import { Pair, clamp, isClose } from "./scale";
import { Transform1D, Transform2D } from "./transform";

import { PlotContext, FigureContext, Axis, styles } from "./plot";


function getScreenCTM(svg: SVGSVGElement): SVGMatrix {
    // workaround on safari
    if (typeof navigator !== 'undefined' && /Version\/[\d\.]+.*Safari/.test(navigator.userAgent)) {
        const m = svg.createSVGMatrix();
        // TODO this doesn't account for viewBox with scaling
        m.e = svg.getBoundingClientRect().x - svg.viewBox.animVal.x;
        m.f = svg.getBoundingClientRect().y - svg.viewBox.animVal.y;
        return m;
    }
    return svg.getScreenCTM()!;
}


function getEventCoords(node: SVGElement, event: MouseEvent | WheelEvent | Touch): Pair {
    let svg = node.ownerSVGElement || node as SVGSVGElement;
    let pt = svg.createSVGPoint();
    pt.x = event.clientX; pt.y = event.clientY;
    let trans = getScreenCTM(svg);
    pt = pt.matrixTransform(trans.inverse());
    return [pt.x, pt.y];
}


export function Zoomer({children: children}: {children?: React.ReactNode}) {
    const fig = React.useContext(FigureContext)!;
    const plot = React.useContext(PlotContext)!;

    const child = React.Children.only(children) as React.ReactElement<React.SVGProps<SVGSVGElement>>;
    const childRef = React.useRef<(HTMLElement & SVGSVGElement) | null>(null);
    const managerRef = React.useRef<ZoomManager | null>(null);

    let xaxis: Axis, yaxis: Axis;
    let xtrans: Transform1D, ytrans: Transform1D;
    let setXTrans: (value: Transform1D) => void, setYTrans: (value: Transform1D) => void;

    if (typeof plot.xaxis === 'string') {
        xaxis = fig.axes.get(plot.xaxis)!;
        [xtrans, setXTrans] = useAtom(fig.transforms.get(plot.xaxis)!)
    } else {
        xaxis = plot.xaxis;
        xtrans = new Transform1D();
        setXTrans = (_: Transform1D) => {};
    }
    if (typeof plot.yaxis === 'string') {
        yaxis = fig.axes.get(plot.yaxis)!;
        [ytrans, setYTrans] = useAtom(fig.transforms.get(plot.yaxis)!)
    } else {
        yaxis = plot.yaxis;
        ytrans = new Transform1D();
        setYTrans = (_: Transform1D) => {};
    }

    React.useEffect(() => {
        if (!managerRef.current) return;
        const manager = managerRef.current;
        manager.setXTrans = setXTrans;
        manager.setYTrans = setYTrans;

        manager.updateTransform(Transform2D.from_1d(xtrans, ytrans));
    }, [xtrans, setXTrans, ytrans, setYTrans]);

    React.useEffect(() => {
        if (!managerRef.current) {
            console.log("Constructing zoomer...");
            let transform = Transform2D.from_1d(xtrans, ytrans);

            managerRef.current = new ZoomManager(
                xaxis, yaxis, transform,
                setXTrans, setYTrans,
                fig.zoomExtent, plot.fixedAspect
            );
        }
        const manager = managerRef.current;

        manager.register(childRef.current!);
        manager.setXTrans = setXTrans;
        manager.setYTrans = setYTrans;
        manager.setTransform(manager.transform);

        return () => {
            manager.unregister(childRef.current!);
        };
    }, [fig, plot.xaxis, plot.yaxis, xaxis, yaxis]);

    return React.cloneElement(child, {ref: childRef});
}

class ZoomManager {
    xaxis: Axis;
    yaxis: Axis;

    transform: Transform2D;
    setXTrans: (val: Transform1D) => void;
    setYTrans: (val: Transform1D) => void;

    zoomExtent: [Pair, Pair]; // [[minx, maxx], [miny, maxy]]
    fixedAspect: boolean;

    state: "drag" | "idle" = "idle";
    dragStart: Pair = [0, 0];
    //transform: Transform2D = new Transform2D();

    elem: (SVGElement & HTMLElement) | null = null;
    listeners: EventListenerManager = new EventListenerManager();
    firstUpdate: boolean = true;

    constructor(
        xaxis: Axis, yaxis: Axis, transform: Transform2D, 
        setXTrans: (val: Transform1D) => void, setYTrans: (val: Transform1D) => void,
        zoomExtent: Pair, fixedAspect: boolean = false
    ) {
        this.xaxis = xaxis;
        this.yaxis = yaxis;

        this.transform = transform;
        this.setXTrans = setXTrans;
        this.setYTrans = setYTrans;

        this.zoomExtent = [zoomExtent, zoomExtent];
        this.fixedAspect = fixedAspect;

        if (fixedAspect) {
            this.setTransform(this.constrainToAspect(this.transform, 'shrink'));
            //this.zoomExtent[0] = [this.zoomExtent[0][0] * this.transform.k[0], this.zoomExtent[0][1] * this.transform.k[0]]
            //this.zoomExtent[1] = [this.zoomExtent[1][0] * this.transform.k[1], this.zoomExtent[1][1] * this.transform.k[1]]
        }
    }

    register(elem: HTMLElement & SVGElement) {
        this.listeners.addEventListener(elem, "mousedown", (ev) => this.mousedown(elem, ev));
        this.listeners.addEventListener(elem, "mousemove", (ev) => this.mousemove(elem, ev));
        this.listeners.addEventListener(elem, "mouseup", (ev) => this.mouseup(elem, ev));
        this.listeners.addEventListener(elem, "wheel", (ev) => this.wheel(elem, ev));
        this.elem = elem;
    }

    unregister(elem: HTMLElement & SVGElement) {
        this.elem = null;
        this.listeners.removeElementListeners(elem);
    }

    setTransform(transform: Transform2D): void {
        this.transform = transform;
        const [xtrans, ytrans] = transform.to_1d();
        this.setXTrans(xtrans); this.setYTrans(ytrans);
    }

    updateTransform(transform: Transform2D) {
        const oldTransform = this.transform;
        this.transform = transform;

        if (this.fixedAspect && !this.firstUpdate) {
            // update y axis if x changed, and vice versa.
            const method = isClose(oldTransform.k[0], transform.k[0]) ? 'x' : 'y';

            transform = this.constrainToAspect(transform, method);
            if (!isClose(this.transform.k, transform.k)) {
                this.transform = transform;
                const [xtrans, ytrans] = transform.to_1d();
                if (method == "x") {
                    this.setXTrans(xtrans);
                } else {
                    this.setYTrans(ytrans);
                }
            }
        }

        if (this.elem) {
            const elems = this.elem.getElementsByClassName(styles['zoom']);
            for (let i = 0; i < elems.length; i++) {
                (elems[i] as SVGElement).setAttribute("transform", this.transform.toString());
            }
        }

        this.firstUpdate = false;
    }

    constrainToAspect(transform: Transform2D, method?: 'x' | 'y' | 'grow' | 'shrink'): Transform2D {
        let kx = Math.abs((this.xaxis.scale.rangeSize() / this.xaxis.scale.linDomainSize()) * this.transform.k[0]);
        let ky = Math.abs((this.yaxis.scale.rangeSize() / this.yaxis.scale.linDomainSize()) * this.transform.k[1]);
        if (isClose(kx, ky)) return transform;

        const [c_x, c_y] = this.transform.unapply([this.xaxis.scale.rangeFromUnit(0.5), this.yaxis.scale.rangeFromUnit(0.5)]);

        let scale: Pair;
        if (method == 'x') {
            scale = [ky/kx, 1.0];
        } else if (method == 'y') {
            scale = [1.0, kx/ky];
        } else if (method == 'grow') {
            scale = [Math.max(1.0, kx/ky), Math.max(1.0, ky/kx)];
        } else { // method == 'shrink'
            scale = [Math.min(1.0, ky/kx), Math.min(1.0, kx/ky)];
        }

        return this.transform.compose(new Transform2D(scale, [c_x*(1-scale[0]), c_y*(1-scale[1])]));
    }

    constrain(transform: Transform2D): Transform2D { 
        if (this.fixedAspect) {
            transform = this.constrainToAspect(transform, 'shrink');
        }

        // taken from d3-zoom
        let currentExtent = [transform.invert().xlim(this.xaxis.scale.range), transform.invert().ylim(this.yaxis.scale.range)];
        // transform translateExtent to range coordinates
        let xExtent = this.xaxis.scale.transform(this.xaxis.translateExtent);
        let yExtent = this.yaxis.scale.transform(this.yaxis.translateExtent);

        // desired shift to bring extent to translateExtent
        let x0 = currentExtent[0][0] - xExtent[0];
        let x1 = currentExtent[0][1] - xExtent[1];
        let y0 = currentExtent[1][0] - yExtent[0];
        let y1 = currentExtent[1][1] - yExtent[1];

        return transform.pretranslate(
            // if x1 > x0, overconstrained, return the average.
            // otherwise, return 
            x1 > x0 ? (x0 + x1) / 2.0 : Math.min(0, x0) + Math.max(0, x1),
            y1 > y0 ? (y0 + y1) / 2.0 : Math.min(0, y0) + Math.max(0, y1),
        );
    }

    mousedown(elem: (HTMLElement & SVGElement), event: MouseEvent) {
        if (event.button != 0) { return; } // LMB only
        const [x, y] = this.transform.unapply(getEventCoords(elem, event));

        this.state = "drag";
        this.dragStart = [x, y];

        this.listeners.addDocumentListener("mousemove", (ev) => this.mousemove(elem, ev));
        this.listeners.addDocumentListener("mouseup", (ev) => this.mouseup(elem, ev));

        event.stopPropagation(); event.preventDefault();
    }

    mousemove(elem: (HTMLElement & SVGElement), event: MouseEvent) {
        if (this.state != "drag") { return; }

        let [x, y] = this.transform.unapply(getEventCoords(elem, event));
        let [deltaX, deltaY] = [x - this.dragStart[0], y - this.dragStart[1]];

        this.setTransform(this.constrain(this.transform.translate(deltaX, deltaY)));
        event.stopPropagation(); event.preventDefault();
    }

    mouseup(elem: (HTMLElement & SVGElement), event: MouseEvent) {
        this.state = "idle";
        this.listeners.removeDocumntListeners();
        event.stopPropagation(); event.preventDefault();
    }

    wheel(elem: (HTMLElement & SVGElement), event: WheelEvent) {
        const [x, y] = getEventCoords(elem, event);
        const k = Math.exp(-event.deltaY / 500.0);
        const totalK: Pair = [
            clamp(k * this.transform.k[0], this.zoomExtent[0]),
            clamp(k * this.transform.k[1], this.zoomExtent[1]),
        ];

        const [origx, origy] = this.transform.unapply([x, y]);
        const transform = new Transform2D(totalK, [-origx * totalK[0] + x, -origy * totalK[1] + y]);

        this.setTransform(this.constrain(transform));
        event.stopPropagation(); event.preventDefault();
    } 
}

class EventListenerManager {
    private elemListeners: Map<HTMLElement, Array<[keyof HTMLElementEventMap, (this: HTMLElement, ev: any) => void, boolean | undefined | AddEventListenerOptions]>> = new Map();
    private docListeners: Array<[keyof DocumentEventMap, (this: Document, ev: any) => void, boolean | undefined | AddEventListenerOptions]> = [];
    private winListeners: Array<[keyof WindowEventMap, (this: Window, ev: any) => void, boolean | undefined | AddEventListenerOptions]> = [];

    addEventListener<K extends keyof HTMLElementEventMap>(
        elem: HTMLElement, type: K,
        listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => void,
        options?: boolean | AddEventListenerOptions,
    ) {
        elem.addEventListener(type, listener, options);

        let arr = this.elemListeners.get(elem);
        if (!arr) {
            this.elemListeners.set(elem, [[type, listener, options]]);
        } else {
            arr.push([type, listener, options]);
        }
    }

    removeElementListeners(elem: HTMLElement) {
        let arr = this.elemListeners.get(elem);
        if (arr) {
            for (const [type, listener, options] of arr) {
                elem.removeEventListener(type, listener, options);
            }
            this.elemListeners.set(elem, []);
        }
    }

    addDocumentListener<K extends keyof DocumentEventMap>(
        type: K, listener: (this: Document, ev: DocumentEventMap[K]) => void,
        options?: boolean | AddEventListenerOptions,
    ) {
        document.addEventListener(type, listener, options);
        this.docListeners.push([type, listener, options]);
    }

    removeDocumntListeners() {
        for (const [type, listener, options] of this.docListeners) {
            document.removeEventListener(type, listener, options);
        }
        this.docListeners = [];
    }

    addWindowListener<K extends keyof WindowEventMap>(
        type: K, listener: (this: Window, ev: WindowEventMap[K]) => void,
        options?: boolean | AddEventListenerOptions,
    ) {
        window.addEventListener(type, listener, options);
        this.winListeners.push([type, listener, options]);
    }

    removeWindowListeners() {
        for (const [type, listener, options] of this.docListeners) {
            window.removeEventListener(type, listener, options);
        }
        this.winListeners = [];
    }
}