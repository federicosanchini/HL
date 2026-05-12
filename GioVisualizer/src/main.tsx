import '@mantine/core/styles.css';
import '@mantine/dropzone/styles.css';

import { MantineProvider } from '@mantine/core';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ThemeProvider } from './context/ThemeContext';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <MantineProvider defaultColorScheme="light">
        <App />
      </MantineProvider>
    </ThemeProvider>
  </StrictMode>,
);
