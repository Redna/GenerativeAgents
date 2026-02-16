import Phaser from 'phaser';

export default {
  type: Phaser.AUTO,
  width: "100%",
  height: "100%",
  parent: "game-container",
  backgroundColor: '#2d2d2d',
  pixelArt: true,
  physics: {
    default: "arcade",
    arcade: {
      gravity: { y: 0 }
    }
  },
  scale: {
    mode: Phaser.Scale.RESIZE,
    autoCenter: Phaser.Scale.CENTER_BOTH
  }
};
