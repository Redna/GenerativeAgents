import { AgentDTO } from "../connector/dtos";
import { MOVEMENT_SPEED, toPixelPosition, toTilePosition } from "../globals";

export class Bubble {
    private bubble: Phaser.GameObjects.Image;
    private emoji: Phaser.GameObjects.Text;
    initials: string;
    plain_emoji: string;

    constructor(initials: string, bubble: Phaser.GameObjects.Image, emoji: Phaser.GameObjects.Text) {
        this.bubble = bubble
        this.emoji = emoji
        this.plain_emoji = emoji.text.split(":")[1].trim()
        this.initials = initials
    }

    setEmoji(emoji: string): void {
        this.emoji.text = this.initials + ":" + emoji
        this.plain_emoji = emoji
    }

    updateX(step: number): void {
        this.bubble.x += step
        this.emoji.x += step
    }

    updateY(step: number): void {
        this.bubble.y += step
        this.emoji.y += step
    }

    getEmoji(): string {
        return this.plain_emoji
    }
}

export default class Character {

    name: string;
    sprite: Phaser.Types.Physics.Arcade.SpriteWithDynamicBody;
    bubble: Bubble;
    movement: { col: number, row: number };
    description: string;
    location: string;
    activity: string;

    // reformat constructor arguments to match DTO
    constructor(name: string, 
                sprite: Phaser.Types.Physics.Arcade.SpriteWithDynamicBody, 
                bubble: Bubble,
                movement: { col: number, row: number },
                description: string, 
                location: string, 
                activity: string) {
        this.name = name;
        this.sprite = sprite;
        this.bubble = bubble;
        this.movement = movement;
        this.description = description;
        this.location = location;
        this.activity = activity;
    }

    position(): { col: number, row: number } {
        return toTilePosition(this.sprite.x, this.sprite.y)
    }

    setEmoji(emoji: string): void {
        this.bubble.setEmoji(emoji)
    }

    update(): void {
        const target = toPixelPosition(this.movement.col, this.movement.row)

        if(this.sprite.x < target.x) {
            this.sprite.anims.play(this.name + "-right-walk", true);
            this.sprite.x += MOVEMENT_SPEED;
            this.bubble.updateX(MOVEMENT_SPEED);
        } else if(this.sprite.x > target.x) {
            this.sprite.anims.play(this.name + "-left-walk", true);
            this.sprite.x -= MOVEMENT_SPEED;
            this.bubble.updateX(-MOVEMENT_SPEED);
        } else if(this.sprite.y < target.y) {
            this.sprite.anims.play(this.name + "-down-walk", true);
            this.sprite.y += MOVEMENT_SPEED;
            this.bubble.updateY(MOVEMENT_SPEED);
        } else if(this.sprite.y > target.y) {
            this.sprite.anims.play(this.name + "-up-walk", true);
            this.sprite.y -= MOVEMENT_SPEED;
            this.bubble.updateY(-MOVEMENT_SPEED);
        } else {
            this.sprite.anims.stop();
        }
    }

    toAgentDto(): AgentDTO {
        return {
            name: this.name,
            description: this.description,
            location: this.location,
            emoji: this.bubble.getEmoji(),
            activity: this.activity,
            movement: {
                col: this.position().col,
                row: this.position().row
            }
        }
    }
}