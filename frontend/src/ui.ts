import { AgentDTO } from "./connector/dtos";

export default class UI {
    private roundElement: HTMLElement | null;
    private timeElement: HTMLElement | null;
    private agentListElement: HTMLElement | null;

    private selectedAgentName: string | null = null;

    constructor() {
        this.roundElement = document.getElementById("status-round");
        this.timeElement = document.getElementById("status-time");
        this.agentListElement = document.getElementById("agent-list");

        const unfocusBtn = document.getElementById("btn-unfocus");
        if (unfocusBtn) {
            unfocusBtn.onclick = () => {
                this.deselectAgent();
            };
        }
    }

    updateStatus(round: number, time: string) {
        if (this.roundElement) this.roundElement.innerText = round.toString();
        if (this.timeElement) this.timeElement.innerText = time;
    }

    updateAgents(agents: AgentDTO[]) {
        if (!this.agentListElement) return;

        // If an agent is selected, update their details live
        if (this.selectedAgentName) {
            const selected = agents.find(a => a.name === this.selectedAgentName);
            if (selected) {
                this.renderAgentDetails(selected);
            }
        }

        this.agentListElement.innerHTML = ""; // Clear list

        agents.forEach(agent => {
            const card = document.createElement("div");
            card.className = "agent-card";
            if (this.selectedAgentName === agent.name) {
                card.style.borderLeftColor = "#ffeb3b"; // Highlight selected in list
                card.style.backgroundColor = "#fff9c4";
            }

            card.onclick = () => {
                this.selectAgent(agent);
            };

            const nameDiv = document.createElement("div");
            nameDiv.className = "agent-name";
            nameDiv.innerText = `${agent.emoji} ${agent.name.replace(/_/g, " ")}`;

            const activityDiv = document.createElement("div");
            activityDiv.className = "agent-activity";
            activityDiv.innerText = agent.activity;

            card.appendChild(nameDiv);
            card.appendChild(activityDiv);
            this.agentListElement?.appendChild(card);
        });
    }

    selectAgent(agent: AgentDTO) {
        this.selectedAgentName = agent.name;
        this.renderAgentDetails(agent);

        // Dispatch event for GameScene to pick up
        window.dispatchEvent(new CustomEvent('agent-selected', { detail: { name: agent.name } }));
    }

    deselectAgent() {
        this.selectedAgentName = null;
        const detailsEl = document.getElementById("agent-details");
        if (detailsEl) {
            detailsEl.style.display = "none";
        }

        // Dispatch event for GameScene to pick up
        window.dispatchEvent(new CustomEvent('deselect-agent'));
    }

    renderAgentDetails(agent: AgentDTO) {
        const DETAILS_ID = "agent-details";
        const detailsEl = document.getElementById(DETAILS_ID);
        if (!detailsEl) return;

        detailsEl.style.display = "block";

        const setText = (id: string, text: string) => {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
        }

        setText("detail-name", agent.name.replace(/_/g, " "));
        setText("detail-age", agent.age ? agent.age.toString() : "N/A");
        setText("detail-desc", agent.description);
        setText("detail-loc", agent.location);
        setText("detail-act", agent.activity);
    }
}
