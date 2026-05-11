import { Stack, Text } from '@mantine/core';
import { Dropzone, MIME_TYPES } from '@mantine/dropzone';

interface Props {
  onFile: (file: File) => void;
}

export default function DropZone({ onFile }: Props) {
  return (
    <Dropzone
      onDrop={(files) => files[0] && onFile(files[0])}
      accept={{ [MIME_TYPES.json]: ['.json'] }}
      maxFiles={1}
    >
      <Stack align="center" gap="xs" py="xl" style={{ pointerEvents: 'none' }}>
        <Text size="xl">Drop backtest JSON here</Text>
        <Text size="sm" c="dimmed">or click to select file</Text>
      </Stack>
    </Dropzone>
  );
}
