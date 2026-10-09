const numericToken = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

// Read CSS's number directly: its string serialization can lose significant digits.
// Create a fresh reader after setting inputs so every value belongs to the same state.
export function cssNumberReader(element) {
  const typed = element.computedStyleMap?.();
  let serialized;
  return (name) => {
    const number = typed?.get(name);
    if (number?.unit === 'number') {
      if (!Number.isFinite(number.value)) throw new Error(`Invalid CSS output: ${name}`);
      return number.value;
    }
    // Browsers without numeric Typed OM retain the existing strict string reader.
    serialized ??= getComputedStyle(element);
    const token = serialized.getPropertyValue(name).trim();
    if (!numericToken.test(token) || !Number.isFinite(Number(token))) throw new Error(`Invalid CSS output: ${name}`);
    return Number(token);
  };
}
