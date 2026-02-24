import { AgentDTO, LogDTO } from './connector/DTOs';
import SimulationConnector from './connector/SimulationConnector';

export default class UI {
  private roundElement: HTMLElement | null;
  private timeElement: HTMLElement | null;
  private agentListElement: HTMLElement | null;
  private logPanel: HTMLElement | null;
  private connector: SimulationConnector;

  private selectedAgentName: string | null = null;

  constructor(connector: SimulationConnector) {
    this.connector = connector;
    this.roundElement = document.getElementById('status-round');
    this.timeElement = document.getElementById('status-time');
    this.agentListElement = document.getElementById('agent-list');

    const unfocusBtn = document.getElementById('btn-unfocus');
    if (unfocusBtn) {
      unfocusBtn.onclick = () => {
        this.deselectAgent();
      };
    }

    const toggleBtn = document.getElementById('btn-toggle-play');
    if (toggleBtn) {
      let isPaused = false;
      toggleBtn.onclick = async () => {
        if (isPaused) {
          await this.connector.resumeSimulation();
          toggleBtn.innerText = '⏸ Pause Engine';
          toggleBtn.style.borderLeftColor = '#ff9800';
          isPaused = false;
        } else {
          await this.connector.pauseSimulation();
          toggleBtn.innerText = '▶ Play Engine';
          toggleBtn.style.borderLeftColor = '#4caf50';
          isPaused = true;
        }
      };
    }

    const xrayBtn = document.getElementById('btn-xray');
    if (xrayBtn) xrayBtn.onclick = () => this.showXRay();

    const closeXrayBtn = document.getElementById('btn-close-xray');
    if (closeXrayBtn) closeXrayBtn.onclick = () => {
      const overlay = document.getElementById('xray-overlay');
      if (overlay) overlay.style.display = 'none';
    };

    this.logPanel = document.getElementById('log-panel');
  }

  updateStatus(round: number, time: string) {
    if (this.roundElement) this.roundElement.innerText = round.toString();
    if (this.timeElement) this.timeElement.innerText = time;
  }

  updateAgents(agents: AgentDTO[]) {
    if (!this.agentListElement) return;

    // If an agent is selected, update their details live
    if (this.selectedAgentName) {
      const selected = agents.find((a) => a.name === this.selectedAgentName);
      if (selected) {
        this.renderAgentDetails(selected);
      }
    }

    this.agentListElement.innerHTML = ''; // Clear list

    agents.forEach((agent) => {
      const card = document.createElement('div');
      card.className = 'agent-card';
      if (this.selectedAgentName === agent.name) {
        card.style.borderLeftColor = '#ffeb3b'; // Highlight selected in list
        card.style.backgroundColor = '#fff9c4';
      }

      card.onclick = () => {
        this.selectAgent(agent);
      };

      const nameDiv = document.createElement('div');
      nameDiv.className = 'agent-name';
      nameDiv.innerText = `${agent.emoji} ${agent.name.replace(/_/g, ' ')}`;

      const activityDiv = document.createElement('div');
      activityDiv.className = 'agent-activity';
      activityDiv.innerText = agent.activity;

      card.appendChild(nameDiv);
      card.appendChild(activityDiv);
      this.agentListElement?.appendChild(card);
    });
  }

  selectAgent(agent: AgentDTO) {
    if (this.selectedAgentName === agent.name) return;

    this.selectedAgentName = agent.name;
    this.renderAgentDetails(agent);
    this.clearLogs();

    // Dispatch event for GameScene to pick up (highlighting)
    // Corrected event name to match Game.ts listener
    window.dispatchEvent(new CustomEvent('agent-subscribe', { detail: { name: agent.name } }));
  }

  deselectAgent() {
    this.selectedAgentName = null;
    this.clearLogs();
    const detailsEl = document.getElementById('agent-details');
    if (detailsEl) {
      detailsEl.style.display = 'none';
    }

    // Dispatch event for GameScene to pick up
    window.dispatchEvent(new CustomEvent('deselect-agent'));
  }

  renderAgentDetails(agent: AgentDTO) {
    const DETAILS_ID = 'agent-details';
    const detailsEl = document.getElementById(DETAILS_ID);
    if (!detailsEl) return;

    detailsEl.style.display = 'block';

    const setText = (id: string, text: string) => {
      const el = document.getElementById(id);
      if (el) el.innerText = text;
    };

    setText('detail-name', agent.name.replace(/_/g, ' '));
    setText('detail-age', agent.age ? agent.age.toString() : 'N/A');
    setText('detail-desc', agent.description);
    setText('detail-loc', agent.location);
    setText('detail-act', agent.activity);
    setText('detail-act', agent.activity);
  }

  renderLogMessage(log: LogDTO) {
    if (!this.logPanel) return;

    // Client-side filtering: only show logs for the selected agent
    if (!this.selectedAgentName) return;
    if (
      this.selectedAgentName !== log.agent &&
      this.selectedAgentName.replace(/_/g, ' ') !== log.agent
    )
      return;

    console.log(log);
    // If it's the first log, clear the "placeholder" text
    if (
      this.logPanel.children.length > 0 &&
      this.logPanel.children[0].tagName === 'DIV' &&
      (this.logPanel.children[0] as HTMLElement).innerText.includes('Select an agent')
    ) {
      this.logPanel.innerHTML = '';
    }

    const logItem = document.createElement('div');
    logItem.style.marginBottom = '5px';
    logItem.style.borderBottom = '1px solid #444';
    logItem.style.paddingBottom = '5px';

    const meta = document.createElement('span');
    meta.style.color = '#888';
    meta.style.fontSize = '0.9em';
    meta.innerText = `${log.timestamp.split(' ')[1]} [${log.level}]: `;

    const msg = document.createElement('span');
    if (log.level === 'DEBUG') msg.style.color = '#aaa';
    else if (log.level === 'WARNING') msg.style.color = '#ffeb3b';
    else if (log.level === 'ERROR') msg.style.color = '#f44336';
    else msg.style.color = '#fff';

    msg.innerText = log.message;

    logItem.appendChild(meta);
    logItem.appendChild(msg);
    this.logPanel.appendChild(logItem);

    // Auto scroll
    this.logPanel.scrollTop = this.logPanel.scrollHeight;
  }

  clearLogs() {
    if (this.logPanel) {
      this.logPanel.innerHTML =
        '<div style="color: #aaa; font-style: italic;">Select an agent to see logs...</div>';
    }
  }

  async showXRay() {
    if (!this.selectedAgentName) return;

    const overlay = document.getElementById('xray-overlay');
    const content = document.getElementById('xray-content');
    const title = document.getElementById('xray-title');
    if (!overlay || !content || !title) return;

    // Show loading state
    title.innerText = `X-Ray: ${this.selectedAgentName.replace(/_/g, ' ')}`;
    content.innerHTML = '<div style="text-align:center; margin-top: 50px; color:#aaa;">Scanning Neural Net...</div>';
    overlay.style.display = 'flex';

    const data = await this.connector.fetchXRay(this.selectedAgentName);
    if (!data) {
      content.innerHTML = '<div style="color:#f44336; padding: 20px;">Error reading agent state from backend.</div>';
      return;
    }

    let html = '';

    // Action Block
    if (data.current_action) {
      html += `
        <div class="xray-section" style="border-left: 4px solid #4a90e2;">
          <div class="xray-label">Active Sequence</div>
          <div style="font-size: 1.1em; color: #fff;">
            ${data.current_action.emoji || ''} ${data.current_action.description || 'Idle'}
          </div>
          <div style="font-size: 0.85em; color: #888; margin-top: 5px;">
            Target: <span style="color: #ccc;">${data.current_action.address || 'N/A'}</span>
          </div>
        </div>
      `;
    }

    // Identity Block
    html += `
      <div class="xray-section">
        <div class="xray-label">Core Persona (System 2 Formulation)</div>
        <div style="color: #ddd;">${data.identity || 'Unformed'}</div>
        <div style="margin-top: 10px; font-size: 0.85em; color: #aaa;">
          Plan: ${data.daily_plan || 'Unplanned'}
        </div>
        <div style="margin-top: 5px; font-size: 0.85em; color: #aaa;">
          Reflection Trigger: <strong>${data.reflection_trigger_counter}</strong> / ${data.reflection_trigger_max}
        </div>
      </div>
    `;

    // Observations Block
    if (data.recent_observations && data.recent_observations.length > 0) {
      html += `<div class="xray-section"><div class="xray-label">Working Memory (Recent Percepts)</div><ul style="margin: 0; padding-left: 20px; color: #ddd;">`;
      data.recent_observations.forEach((obs: string) => {
        html += `<li style="margin-bottom: 4px;">${obs}</li>`;
      });
      html += `</ul></div>`;
    }

    // Graph Memory Block
    if (data.core_memories && data.core_memories.length > 0) {
      html += `<div class="xray-section"><div class="xray-label">Relevant Deep Memories (Qdrant Semantic Recall)</div><ul style="margin: 0; padding-left: 20px; color: #ddd;">`;
      data.core_memories.forEach((mem: string) => {
        html += `<li style="margin-bottom: 4px; font-style: italic;">${mem}</li>`;
      });
      html += `</ul></div>`;
    }

    content.innerHTML = html;
  }
}
