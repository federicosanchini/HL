import { Stack, Text } from "@mantine/core";
import { Dropzone } from "@mantine/dropzone";

interface Props {
  onFile: (file: File) => void;
}

export default function DropZone({ onFile }: Props) {
  return (
    <Dropzone
      onDrop={(files) => files[0] && onFile(files[0])}
      accept={{ "application/json": [".json"] }}
      maxFiles={1}
      styles={{ root: { background: "transparent", border: "none" } }}
    >
      <Stack align="center" gap="xs" py="xl" style={{ pointerEvents: "none" }}>
        <Text size="xl" style={{ color: "var(--text-primary)" }}>
          Drop backtest JSON here
        </Text>
        <Text size="sm" style={{ color: "var(--text-secondary)" }}>
          or click to select file
        </Text>
      </Stack>
    </Dropzone>
  );
}
