import React from "react";
import { ActionIcon, Container, Group, Modal, Text, useMantineColorScheme } from "@mantine/core";
import TimeAgo from "react-timeago";

import { IconSettings, IconBrightnessFilled, IconInfoCircle, IconBrandGithub } from "@tabler/icons-react";
import PhaserLogoText from "../static/logo-text.svg";
import PhaserLogo from "../static/logo.svg";
import { useDisclosure } from "@mantine/hooks";
import { useAtomValue, PrimitiveAtom } from "jotai";
import { ConnectionStatus } from "./connection";
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

// `actions` are page-specific icons, prepended to the shared ones (the dashboard's palette
// and split toggles).
export default function Header({serverStatus, actions}: {serverStatus: PrimitiveAtom<ConnectionStatus>, actions?: React.ReactNode}) {
    const { colorScheme, toggleColorScheme } = useMantineColorScheme();
    const flip = colorScheme === "dark" ? {style: {transform: "scale(-1)"}} : {};

    const [aboutOpened, { open: openAbout, close: closeAbout }] = useDisclosure(false);
    const status = useAtomValue(serverStatus);

    return <Container size="lg" h="100%" style={{minWidth: "500px"}}>
        <Group justify="space-between" h="100%" wrap="nowrap">
            <Group style={{height: "100%"}}>
                <a style={{height: "95%"}} href="/">
                    <PhaserLogoText className="mantine-visible-from-sm" height="100%"/>
                    <PhaserLogo className="mantine-hidden-from-sm" height="100%"/>
                </a>
                <Text size="lg"><StatusText status={status}/></Text>
            </Group>
            <Group wrap="nowrap">
                {actions}
                <ActionIcon size="xl" aria-label="Toggle dark mode" onClick={toggleColorScheme}><IconBrightnessFilled size="80%" {...flip}/></ActionIcon>
                <ActionIcon size="xl" aria-label="Settings"><IconSettings size="80%"/></ActionIcon>
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

    React.useEffect(() => {
        if (!opened || version) return;
        let cancelled = false;
        fetch(`${rootPrefix()}/version`)
            .then((resp) => resp.json())
            .then((info: VersionInfo) => { if (!cancelled) setVersion(info); })
            .catch((e) => console.error("Couldn't fetch version:", e));
        return () => { cancelled = true; };
    }, [opened, version]);

    return <Modal opened={opened} onClose={close} title="About phaser">
        <Text>{version ? versionText(version) : "\u00A0"}</Text>
    </Modal>
}