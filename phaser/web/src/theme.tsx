import { AppShell, AppShellStylesNames, Button, ButtonStylesNames, createTheme, Tabs, TabsStylesNames } from "@mantine/core"
import tabs_classes from './Tabs.module.css'
import appshell_classes from './AppShell.module.css'
import button_classes from './Button.module.css'

export const makeTheme = () => createTheme({
    //fontFamily: 'Open Sans, sans-serif',
    components: {
        Tabs: Tabs.extend({
            classNames: tabs_classes as Partial<Record<TabsStylesNames, string>>,
        }),
        AppShell: AppShell.extend({
            classNames: appshell_classes as Partial<Record<AppShellStylesNames, string>>,
        }),
        Button: Button.extend({
            classNames: button_classes as Partial<Record<ButtonStylesNames, string>>,
        }),
    },
    defaultRadius: 'md',

    primaryColor: "blue",
    colors: {
        dark: [
            '#e2daeb',  // text color
            '#b8b8b8',
            '#828282',
            '#696969',  // placeholder, disabled color
            '#424242',  // border
            '#3b3b3b',  // hover
            '#2e2e2e',  // disabled
            '#251f1f',  // bg
            '#1f1f1f',  // dark filled
            '#141414',  // dark filled hover
        ],
    },
    //--mantine-color-default-border
});

export const cssVariableResolver = (theme: ReturnType<typeof makeTheme>) => ({
    variables: {
    },
    light: {
    },
    dark: {
    },
});