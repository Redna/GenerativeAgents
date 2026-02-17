import Phaser from 'phaser';
import config from './config';
import GameScene from './scenes/Game';
import SimulationConnector from './connector/simulationConnector';


new Phaser.Game(
  Object.assign(config, {
    scene: [GameScene]
  })
);

