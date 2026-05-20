import { useState, useCallback, useEffect } from 'react';

export function useGraphViewport() {
  const [graphContainerEl, setGraphContainerEl] = useState<HTMLDivElement | null>(null);
  const [graphViewport, setGraphViewport] = useState({ width: 0, height: 0 });

  const handleGraphContainerRef = useCallback((node: HTMLDivElement | null) => {
    setGraphContainerEl(node);
  }, []);

  useEffect(() => {
    const container = graphContainerEl;
    if (!container) return;

    const updateViewport = () => {
      setGraphViewport({
        width: container.clientWidth,
        height: container.clientHeight,
      });
    };

    updateViewport();
    const resizeObserver = new ResizeObserver(updateViewport);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, [graphContainerEl]);

  return { graphViewport, handleGraphContainerRef };
}
