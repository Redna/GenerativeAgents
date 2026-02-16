import { io, Socket } from "socket.io-client"
import { AgentDTO, RoundUpdateDTO } from "./dtos";
import Character from "../scenes/Character";

interface ServerToClientEvents {
    update: (roundUpdate: RoundUpdateDTO) => void;
}

interface ClientToServerEvents {
    watch: () => void;
    spawn: (agent: AgentDTO) => void;
}

export default class SimulationConnector {

    private socket: Socket<ServerToClientEvents, ClientToServerEvents>;

    // constructor
    constructor() {
        this.socket = io("http://localhost:8000", { reconnectionDelay: 240000, reconnectionAttempts: 10 });
        this.socket.emit("watch")
    }

    spawn(character: Character) {
        this.socket.emit("spawn", character.toAgentDto())
    }

    onUpdate(callback: (roundUpdate: RoundUpdateDTO) => void) {
        this.socket.on("update", (roundUpdate: RoundUpdateDTO) => {
            // Seems like a bug of socket.io that the payload is not properly parsed...
            console.log(roundUpdate)
            if (typeof roundUpdate === "string") {
                callback(JSON.parse(roundUpdate));
            }
            else {
                callback(roundUpdate);
            }
        });
    }

}
