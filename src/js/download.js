export function downloadDataset(dataset) {
  const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = dataset.modelProfile === 'lookahead-616' ? 'training-data-616.json' : 'training-data-516.json';
  document.body.append(link);
  link.click();
  link.remove();
  // Keep the URL alive long enough for the browser to begin its download.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
