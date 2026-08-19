import React from "react";
import { ActionIcon, Container, Group, Modal, Text, useMantineColorScheme } from "@mantine/core";
import TimeAgo from "react-timeago";

import { IconSettings, IconBrightnessFilled, IconInfoCircle, IconBrandGithub } from "@tabler/icons-react";
import PhaserLogoText from "../static/logo-text.svg";
import PhaserLogo from "../static/logo.svg";
import { useDisclosure } from "@mantine/hooks";
import { useAtomValue, PrimitiveAtom } from "jotai";
import { ConnectionStatus } from "./connection";
import { useGetAction } from "./requests";
import { rootPrefix } from "./utils";


function StatusText({status}: {status: ConnectionStatus}) {
    switch (status.type) {
        case 'connecting': return <>Connecting…</>;
        case 'connected': return <>Connected</>;
        case 'reconnecting': return <>Reconnecting <TimeAgo date={status.retryAt}/></>;
        case 'disconnected': return <>Disconnected</>;
        case 'stopped': return <>Server stopped — refresh to reconnect</>;
    }
}

// split out so `Header` can omit it entirely: pages without a websocket (the error page)
// pass no atom, and a conditional `useAtomValue` in `Header` would break the hook rules.
function ServerStatus({serverStatus}: {serverStatus: PrimitiveAtom<ConnectionStatus>}) {
    return <Text size="lg"><StatusText status={useAtomValue(serverStatus)}/></Text>;
}

// `actions` are page-specific icons, prepended to the shared ones (the dashboard's palette
// and split toggles). `serverStatus` is absent on pages with no connection to report.
// `info` is whatever the page is *about* -- the dashboard passes its job's state; the
// manager and error page pass nothing.
export default function Header({serverStatus, info, actions, size}: {
    serverStatus?: PrimitiveAtom<ConnectionStatus>,
    info?: React.ReactNode,
    actions?: React.ReactNode,
    size?: string | number
}) {
    const { colorScheme, toggleColorScheme } = useMantineColorScheme();
    const flip = colorScheme === "dark" ? {style: {transform: "scale(-1)"}} : {};

    const [aboutOpened, { open: openAbout, close: closeAbout }] = useDisclosure(false);

    return <Container size={size ?? "lg"} h="100%" style={{minWidth: "500px"}}>
        <Group justify="space-between" h="100%" wrap="nowrap">
            <Group style={{height: "100%"}} wrap="nowrap">
                <a style={{height: "95%"}} href={`${rootPrefix()}/`}>
                    <PhaserLogoText className="mantine-visible-from-sm" height="100%"/>
                    <PhaserLogo className="mantine-hidden-from-sm" height="100%"/>
                </a>
                {serverStatus && <ServerStatus serverStatus={serverStatus}/>}
                {info}
            </Group>
            <Group wrap="nowrap">
                {actions}
                <ActionIcon size="xl" aria-label="Toggle dark mode" onClick={toggleColorScheme}><IconBrightnessFilled size="80%" {...flip}/></ActionIcon>
                {/* <ActionIcon size="xl" aria-label="Settings"><IconSettings size="80%"/></ActionIcon> */}
                <About opened={aboutOpened} close={closeAbout}/>
                <ActionIcon size="xl" aria-label="About" onClick={openAbout}><IconInfoCircle size="80%"/></ActionIcon>
                <ActionIcon component="a" classNames={{root: "disable-active"}} href="https://github.com/hexane360/phaser" size="xl" aria-label="GitHub"><IconBrandGithub size="80%"/></ActionIcon>
            </Group>
        </Group>
    </Container>
}

interface VersionInfo {
    version: string,
    git: {commit: string, short_commit: string, dirty: boolean, branch: string | null} | null,
}

function versionText({version, git}: VersionInfo): string {
    if (!git) return `Version ${version}`;
    const parts = [git.branch, git.short_commit, git.dirty ? "dirty" : null].filter((p) => p);
    return `Version ${version} (${parts.join(", ")})`;
}

export function About({opened, close}: {opened: boolean, close: () => void}) {
    /* TODO:
     - brief instructions
     - link to documentation
     - acknowledgments
     */
    const [version, setVersion] = React.useState<VersionInfo | null>(null);
    // quiet: nothing was clicked, so this belongs in the modal rather than over the page
    const [fetchVersion, , error] = useGetAction<VersionInfo>("Couldn't load version", {quiet: true});

    // One attempt per opening: a failure leaves `version` null, so without the latch the
    // effect would re-fire the moment `pending` cleared and retry forever. Reopening the
    // modal is how you ask again.
    const attempted = React.useRef(false);
    React.useEffect(() => {
        if (!opened) {
            attempted.current = false;
            return;
        }
        if (version || attempted.current) return;
        attempted.current = true;
        fetchVersion(`${rootPrefix()}/version`).then((info) => { if (info) setVersion(info); });
    }, [opened, version, fetchVersion]);

    return <Modal opened={opened} onClose={close} title="About phaser">
        <Text c={error ? "red" : undefined}>{version ? versionText(version) : error ?? "\u00A0"}</Text>
    </Modal>
}