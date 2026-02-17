import { io, Socket } from 'socket.io-client';
import { AgentDTO, RoundUpdateDTO, LogDTO } from './DTOs';
import Character from '../scenes/Character';

interface ServerToClientEvents {
  update: (roundUpdate: RoundUpdateDTO) => void;
  agent_log: (log: LogDTO | string) => void;
}

interface ClientToServerEvents {
  watch: () => void;
  spawn: (agent: AgentDTO) => void;
  subscribe_agent_log: (data: { agent_name: string }) => void;
  unsubscribe_agent_log: (data: { agent_name: string }) => void;
}

export default class SimulationConnector {
  private socket: Socket<ServerToClientEvents, ClientToServerEvents>;

  // constructor
  constructor() {
    console.log('Connector: Initializing socket...');
    this.socket = io('http://localhost:8000', {
      reconnectionDelay: 240000,
      reconnectionAttempts: 10,
    });

    this.socket.on('connect', () => {
      console.log('Connector: Socket connected!', this.socket.id);
    });

    this.socket.on('connect_error', (err) => {
      console.error('Connector: Connection error', err);
    });

    this.socket.emit('watch');
  }

  spawn(character: Character) {
    this.socket.emit('spawn', character.toAgentDto());
  }

  onUpdate(callback: (roundUpdate: RoundUpdateDTO) => void) {
    this.socket.on('update', (roundUpdate: RoundUpdateDTO) => {
      // Seems like a bug of socket.io that the payload is not properly parsed...
      if (typeof roundUpdate === 'string') {
        callback(JSON.parse(roundUpdate));
      } else {
        callback(roundUpdate);
      }
    });
  }

  onAgentLog(callback: (log: LogDTO) => void) {
    console.log('Connector: Setting up agent_log listener');
    this.socket.on('agent_log', (log: LogDTO | string) => {
      console.log('Connector: RAW LISTEN agent_log', log);
      let parsedLog: LogDTO;
      if (typeof log === 'string') {
        try {
          parsedLog = JSON.parse(log);
        } catch (e) {
          console.error('Connector: Failed to parse agent_log', e);
          return;
        }
      } else {
        parsedLog = log;
      }
      console.log('Connector: Received agent_log parsed', parsedLog);
      callback(parsedLog);
    });
  }

  subscribeAgentLog(agentName: string) {
    console.log('Connector: Emitting subscribe_agent_log for', agentName);
    this.socket.emit('subscribe_agent_log', { agent_name: agentName });
    console.log('Connector: Emit called.');
  }
}
