declare module '*.module.css' {
    const classes: {readonly [key: string]: string};
    export default classes;
}
declare module '*.css' {}

declare module "*.svg?url" {
    const url: string;
    export default url;
}

declare module "*.svg" {
    import React from "react";
    const ReactComponent: React.FC<React.SVGProps<SVGSVGElement>>;
    export default ReactComponent;
}