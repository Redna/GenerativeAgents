import Phaser from 'phaser';
import Character, { Bubble } from './Character';
import SimulationConnector from '../connector/simulationConnector';
import { MOVEMENT_SPEED, UPDATE_INTERVAL_MS, toPixelPosition } from '../globals';
import { RoundUpdateDTO } from '../connector/dtos';
import UI from '../ui';

class SimulationUpdateEngine {
  private updates: RoundUpdateDTO[];
  private pointer: number;
  private updateInterval: number = 1_200;
  private interval: ReturnType<typeof setInterval> | undefined;
  private characters: { [key: string]: Character };
  private connector: SimulationConnector;
  private gameScene: GameScene;
  private ui: UI;

  constructor(GameScene: GameScene) {
    this.updates = [];
    this.pointer = 0;
    this.characters = {};
    this.gameScene = GameScene;
    this.ui = new UI();

    this.connector = new SimulationConnector()
    this.connector.onUpdate((update: RoundUpdateDTO) => {
      this.add(update);
    });
  }

  addCharacters(characters: { [key: string]: Character }): void {
    for (let name in characters) {
      this.addCharacter(characters[name])
    }
  }

  addCharacter(character: Character): void {
    this.characters[character.name] = character;
  }

  add(update: RoundUpdateDTO): void {
    this.updates.push(update);
  }

  hasNext(): boolean {
    return this.pointer < this.updates.length;
  }

  setPointer(pointer: number): void {
    if (pointer < this.updates.length) {
      this.pointer = pointer;
    }
  }

  getPointer(): number {
    return this.pointer;
  }

  start(): void {
    this.interval = setInterval(() => {
      if (this.hasNext()) {
        this.applyUpdate();
        this.pointer++;
      }
    }, this.updateInterval);
  }

  pause(): void {
    if (this.interval) {
      clearInterval(this.interval);
    }
  }

  private applyUpdate(): void {
    const update = this.updates[this.pointer];
    this.ui.updateStatus(update.round, update.time);
    this.ui.updateAgents(update.agents);

    for (let agent of update.agents) {
      console.log(agent)
      let agent_name = agent.name.replace(" ", "_")
      if (agent_name in this.characters) {

        const character = this.characters[agent_name]
        character.movement = agent.movement
        character.setEmoji(agent.emoji)
        character.description = agent.description
        character.location = agent.location
        character.activity = agent.activity
      }
      else {
        this.gameScene.spawnSprite(agent_name, agent.movement.col, agent.movement.row)
      }
    }
  }
}


export default class GameScene extends Phaser.Scene {

  private character_names: { [key: string]: number[] } = {
    /*"Klaus_Mueller": [127, 46],
    "Maria_Lopez": [127, 54],
    "Tom_Moreno": [73, 14],*/
  }


  private map: Phaser.Tilemaps.Tilemap | undefined

  private characters: any = {};
  private player: Phaser.Types.Physics.Arcade.SpriteWithDynamicBody | undefined;


  private simulationUpdateEngine: SimulationUpdateEngine;

  constructor() {
    super('GameScene');
    this.simulationUpdateEngine = new SimulationUpdateEngine(this)
  }

  preload() {
    this.load.tilemapTiledJSON("map", "./assets/the_ville/visuals/the_ville_jan7.json");
    this.load.image("blocks_1", "assets/the_ville/visuals/map_assets/blocks/blocks_1.png");
    this.load.image("blocks_2", "assets/the_ville/visuals/map_assets/blocks/blocks_2.png");
    this.load.image("blocks_3", "assets/the_ville/visuals/map_assets/blocks/blocks_3.png");
    this.load.image("walls", "assets/the_ville/visuals/map_assets/v1/Room_Builder_32x32.png");
    this.load.image("interiors_pt1", "assets/the_ville/visuals/map_assets/v1/interiors_pt1.png");
    this.load.image("interiors_pt2", "assets/the_ville/visuals/map_assets/v1/interiors_pt2.png");
    this.load.image("interiors_pt3", "assets/the_ville/visuals/map_assets/v1/interiors_pt3.png");
    this.load.image("interiors_pt4", "assets/the_ville/visuals/map_assets/v1/interiors_pt4.png");
    this.load.image("interiors_pt5", "assets/the_ville/visuals/map_assets/v1/interiors_pt5.png");
    this.load.image("CuteRPG_Field_B", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_B.png");
    this.load.image("CuteRPG_Field_C", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Field_C.png");
    this.load.image("CuteRPG_Harbor_C", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Harbor_C.png");
    this.load.image("CuteRPG_Village_B", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Village_B.png");
    this.load.image("CuteRPG_Forest_B", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Forest_B.png");
    this.load.image("CuteRPG_Desert_C", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Desert_C.png");
    this.load.image("CuteRPG_Mountains_B", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Mountains_B.png");
    this.load.image("CuteRPG_Desert_B", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Desert_B.png");
    this.load.image("CuteRPG_Forest_C", "assets/the_ville/visuals/map_assets/cute_rpg_word_VXAce/tilesets/CuteRPG_Forest_C.png");

    this.load.atlas("atlas", "./assets/characters/Yuriko_Yamamoto.png",
      "./assets/characters/atlas.json");


    let character_files = [
      "Abigail_Chen", "Adam_Smith", "Arthur_Burton", "Ayesha_Khan", "Carlos_Gomez",
      "Carmen_Ortiz", "Eddy_Lin", "Francisco_Lopez", "Giorgio_Rossi", "Hailey_Johnson",
      "Isabella_Rodriguez", "Jane_Moreno", "Jennifer_Moore", "John_Lin", "Klaus_Mueller",
      "Latoya_Williams", "Maria_Lopez", "Mei_Lin", "Rajiv_Patel", "Ryan_Park",
      "Sam_Moore", "Tamara_Taylor", "Tom_Moreno", "Wolfgang_Schulz", "Yuriko_Yamamoto"
    ]
    character_files.forEach(character_name => {
      const character_path = "./assets/characters/" + character_name + ".png";
      this.load.atlas(character_name, character_path, "./assets/characters/atlas.json")
    });

    this.load.image('speech_bubble', "./assets/speech_bubble/v3.png");
  }

  private loadTileset(name: string, image_name: string): Phaser.Tilemaps.Tileset {
    const tileset = this.map!.addTilesetImage(name, image_name);

    if (tileset == null) {
      throw Error("Tileset with name " + name + " not found!!")
    }

    return tileset
  }

  create() {
    this.map = this.make.tilemap({ key: "map" });

    const tilesetGroup: Phaser.Tilemaps.Tileset[] = []
    const collisions: Phaser.Tilemaps.Tileset[] = []
    const walls: Phaser.Tilemaps.Tileset[] = []


    collisions.push(this.loadTileset("blocks", "blocks_1"))
    collisions.push(this.loadTileset("blocks_2", "blocks_2"))
    collisions.push(this.loadTileset("blocks_3", "blocks_3"))
    walls.push(this.loadTileset("Room_Builder_32x32", "walls"))

    const mixedTileset = this.loadTileset("CuteRPG_Field_C", "CuteRPG_Field_C")

    walls.push(mixedTileset)

    tilesetGroup.push(mixedTileset)
    tilesetGroup.push(this.loadTileset("interiors_pt1", "interiors_pt1"))
    tilesetGroup.push(this.loadTileset("interiors_pt2", "interiors_pt2"))
    tilesetGroup.push(this.loadTileset("interiors_pt3", "interiors_pt3"))
    tilesetGroup.push(this.loadTileset("interiors_pt4", "interiors_pt4"))
    tilesetGroup.push(this.loadTileset("interiors_pt5", "interiors_pt5"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Field_B", "CuteRPG_Field_B"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Harbor_C", "CuteRPG_Harbor_C"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Village_B", "CuteRPG_Village_B"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Forest_B", "CuteRPG_Forest_B"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Desert_C", "CuteRPG_Desert_C"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Mountains_B", "CuteRPG_Mountains_B"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Desert_B", "CuteRPG_Desert_B"))
    tilesetGroup.push(this.loadTileset("CuteRPG_Forest_C", "CuteRPG_Forest_C"))

    this.map.createLayer("Bottom Ground", tilesetGroup, 0, 0);
    this.map.createLayer("Exterior Ground", tilesetGroup, 0, 0);
    this.map.createLayer("Exterior Decoration L1", tilesetGroup, 0, 0);
    this.map.createLayer("Exterior Decoration L2", tilesetGroup, 0, 0);
    this.map.createLayer("Interior Ground", tilesetGroup, 0, 0);
    this.map.createLayer("Wall", walls, 0, 0);
    this.map.createLayer("Interior Furniture L1", tilesetGroup, 0, 0);
    this.map.createLayer("Interior Furniture L2 ", tilesetGroup, 0, 0);

    const foregroundL1Layer = this.map.createLayer("Foreground L1", tilesetGroup, 0, 0);
    const foregroundL2Layer = this.map.createLayer("Foreground L2", tilesetGroup, 0, 0);
    const collisionsLayer = this.map.createLayer("Collisions", collisions, 0, 0);

    collisionsLayer!.setCollisionByProperty({ collide: true });
    // By default, everything gets depth sorted on the screen in the order we 
    // created things. Here, we want the "Above Player" layer to sit on top of 
    // the player, so we explicitly give it a depth. Higher depths will sit on 
    // top of lower depth objects.
    // Collisions layer should get a negative depth since we do not want to see
    // it. 
    collisionsLayer!.setDepth(-1);
    foregroundL1Layer!.setDepth(2);
    foregroundL2Layer!.setDepth(2);

    this.player = this.physics.add.sprite(2400, 588, "atlas", "down").setSize(30, 40).setOffset(0, 0);

    this.player.setDepth(-1);
    // Setting up the camera. 
    const camera = this.cameras.main;

    // Center on the map
    camera.setBounds(0, 0, this.map.widthInPixels, this.map.heightInPixels);

    const zoomX = this.sys.game.canvas.width / this.map.widthInPixels;
    const zoomY = this.sys.game.canvas.height / this.map.heightInPixels;
    const minZoom = Math.min(zoomX, zoomY);

    camera.setZoom(minZoom);
    camera.centerOn(this.map.widthInPixels / 2, this.map.heightInPixels / 2);

    // Mouse Navigation
    this.input.on('pointerdown', (pointer: Phaser.Input.Pointer) => {
      if (pointer.isDown) {
        this.input.mouse?.requestPointerLock();
        this.cameras.main.stopFollow(); // Stop following if user takes control
      }
    });

    this.input.on('pointermove', (pointer: Phaser.Input.Pointer) => {
      if (this.input.mouse?.locked) {
        camera.scrollX -= pointer.movementX / camera.zoom;
        camera.scrollY -= pointer.movementY / camera.zoom;
      }
    });

    this.input.on('pointerup', () => {
      this.input.mouse?.releasePointerLock();
    });

    // Zoom
    this.input.on('wheel', (pointer: Phaser.Input.Pointer, gameObjects: any, deltaX: number, deltaY: number, deltaZ: number) => {
      const newZoom = camera.zoom - deltaY * 0.001;
      // Limit zoom
      camera.setZoom(Phaser.Math.Clamp(newZoom, 0.1, 2));
    });

    // Listen for UI selection events
    window.addEventListener('agent-selected', (e: any) => {
      const name = e.detail.name;
      this.highlightAgent(name);
    });

    window.addEventListener('deselect-agent', () => {
      this.selectedAgentName = null;
      this.cameras.main.stopFollow();
      if (this.selectionGraphics) {
        this.selectionGraphics.clear();
      }
    });

    for (let name in this.character_names) {
      this.spawnSprite(name, this.character_names[name][0], this.character_names[name][1])
    }

    this.simulationUpdateEngine.start();
  }

  private selectionGraphics: Phaser.GameObjects.Graphics | undefined;
  private selectedAgentName: string | null = null;

  private highlightAgent(name: string) {
    this.selectedAgentName = name.replace(" ", "_");

    const character = this.characters[this.selectedAgentName];
    if (character && character.sprite) {
      // Smoothly follow the selected agent
      this.cameras.main.startFollow(character.sprite, true, 0.08, 0.08);
    }

    // Force immediate update so we don't wait for next frame
    this.updateHighlight(this.game.getTime());
  }

  private updateHighlight(time: number = 0) {
    if (!this.selectedAgentName) return;

    if (!this.selectionGraphics) {
      // Depth 0.5 ensures it is above ground (0) but below agents (1)
      this.selectionGraphics = this.add.graphics().setDepth(0.5);
    }
    this.selectionGraphics.clear();

    const character = this.characters[this.selectedAgentName];
    if (character && character.sprite) {
      const sprite = character.sprite;

      // Pulsing effect: Alpha oscillates between 0.3 and 0.8 over time
      const pulseSpeed = 0.005;
      const alpha = 0.55 + 0.25 * Math.sin(time * pulseSpeed);

      // FIFA-style marker: Red circle/ellipse under feet
      this.selectionGraphics.lineStyle(4, 0xff0000, alpha);
      this.selectionGraphics.fillStyle(0xff0000, alpha * 0.4);

      const feetY = sprite.y + (sprite.displayHeight / 2) - 2;

      this.selectionGraphics.fillEllipse(sprite.x, feetY, 44, 22);
      this.selectionGraphics.strokeEllipse(sprite.x, feetY, 44, 22);
    }
  }

  update(time: number, delta: number): void {
    // iterate through characters and update their position 
    for (let name in this.characters) {
      const character = this.characters[name]
      character.update()
    }

    if (this.selectedAgentName) {
      this.updateHighlight(time);
    }
  }

  public spawnSprite(name: string, cellX: number, cellY: number): Character {
    let position = toPixelPosition(cellX, cellY);

    let sprite = this.physics.add
      .sprite(position.x, position.y, name, "down")
      .setSize(30, 40)
      .setOffset(0, 0)
      .setDepth(1); // Ensure sprite is above the highlight (depth 0.5)

    sprite.displayWidth = 40;
    sprite.scaleY = sprite.scaleX;

    let bubble = this.add.image(position.x + 60, position.y - 40, 'speech_bubble').setDepth(3);
    bubble.displayWidth = 110;
    bubble.displayHeight = 50;

    const textStyle = {
      font: "28px monospace",
      color: "#000000",
      padding: { x: 8, y: 8 }
    }

    let emoji = this.add.text(position.x + 5, position.y - 67, this.getInitials(name) + ":🦁", textStyle).setDepth(3);

    this.createSpriteAnimation(name);

    const character = new Character(name, sprite, new Bubble(this.getInitials(name), bubble, emoji), { col: cellX, row: cellY }, "description", "", "");
    this.simulationUpdateEngine.addCharacter(character);
    this.character_names[name] = [cellX, cellY];
    this.characters[name] = character;
    return character;
  }

  private getInitials(name: string): string {
    const rgx = /(\p{L}{1})\p{L}+/gu;
    let matches = [...name.matchAll(rgx)];
    let initials = ((matches.shift()?.[1] || '') + (matches.pop()?.[1] || '')).toUpperCase();
    return initials
  }

  private createSpriteAnimation(name: string) {
    let left_walk_name = name + "-left-walk";
    let right_walk_name = name + "-right-walk";
    let down_walk_name = name + "-down-walk";
    let up_walk_name = name + "-up-walk";

    console.log(name, left_walk_name, "DEUBG")
    this.anims.create({
      key: left_walk_name,
      frames: this.anims.generateFrameNames(name, { prefix: "left-walk.", start: 0, end: 3, zeroPad: 3 }),
      frameRate: 4,
      repeat: -1
    });

    this.anims.create({
      key: right_walk_name,
      frames: this.anims.generateFrameNames(name, { prefix: "right-walk.", start: 0, end: 3, zeroPad: 3 }),
      frameRate: 4,
      repeat: -1
    });

    this.anims.create({
      key: down_walk_name,
      frames: this.anims.generateFrameNames(name, { prefix: "down-walk.", start: 0, end: 3, zeroPad: 3 }),
      frameRate: 4,
      repeat: -1
    });

    this.anims.create({
      key: up_walk_name,
      frames: this.anims.generateFrameNames(name, { prefix: "up-walk.", start: 0, end: 3, zeroPad: 3 }),
      frameRate: 4,
      repeat: -1
    });
  }

}
