
export interface MovementDTO {
    col: number;
    row: number;
}

export interface AgentDTO {
    name: string;
    age: number;
    inniate_traits: string[];
    description: string;
    location: string;
    emoji: string;
    activity: string;
    movement: MovementDTO;
}

export interface LogDTO {
    agent: string;
    message: string;
    level: string;
    timestamp: string;
    tick: number;
}

export interface RoundUpdateDTO {
    agents: AgentDTO[]
    round: number
    time: string
}