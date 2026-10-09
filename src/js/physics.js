export function updateBird(bird, deltaSeconds, config) {
  bird.velocityY += config.gravity * deltaSeconds;
  bird.y += bird.velocityY * deltaSeconds;
}

export function circleHitsRectangle(x, y, radius, rectangle) {
  const nearestX = Math.max(rectangle.x, Math.min(x, rectangle.x + rectangle.width));
  const nearestY = Math.max(rectangle.y, Math.min(y, rectangle.y + rectangle.height));
  return (x - nearestX) ** 2 + (y - nearestY) ** 2 <= radius ** 2;
}
