const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const B=(t,c)=>`<span class="b ${c||''}">${t}</span>`;
const SEV={Critical:'r',High:'a',Medium:'p',Low:'g',Info:'c'},ST={Completed:'g',Active:'g',Processing:'c',"Needs Review":'a',Archived:'',"Under Review":'a',Proposed:'c',Superseded:'',Reversed:'r',Blocked:'r',Overdue:'r',"In Progress":'c',"Not Started":'',Open:'r',Mitigating:'a',Ready:'g',Draft:'a',Scheduled:'c',Success:'g'};
const bs=t=>B(t,ST[t]||SEV[t]);
const TB=(h,r)=>`<div class="tw"><table><thead><tr>${h.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${r.join('')}</tbody></table></div>`;
const PH=(t,s)=>`<h2 style="font-size:20px">${t}</h2><p class="sub">${s}</p>`;
const toast=t=>{const e=$('#toast');e.textContent=t;e.className='on';setTimeout(()=>e.className='',1800)};
const bar=p=>`<div class="bar"><i style="width:${p}%"></i></div>`;
const kpi=(n,v,tr,ic)=>`<div class="card kpi"><small>${ic} ${n}</small><b>${v}</b><span class="${tr[0]=='-'?'dn':'up'}">${tr}</span> <small>vs prior 30d</small></div>`;
const tabs=(id,l,cur)=>`<div class="tabs" data-tabs="${id}">${l.map(x=>`<button class="${x==cur?'on':''}" data-t="${x}">${x}</button>`).join('')}</div>`;
let timer=null,S={meet:0,tab:{},view:'List',acts:ACT.map(a=>[...a])};

function area(a,b){const W=560,H=170,mx=Math.max(...a,...b);const pt=v=>v.map((y,i)=>`${i*W/(v.length-1)},${H-y/mx*(H-10)}`).join(' ');
return `<svg viewBox="0 0 ${W} ${H+16}" width="100%"><polygon points="0,${H} ${pt(b)} ${W},${H}" fill="#38bdf833"/><polyline pathLength="1" points="${pt(b)}" fill="none" stroke="#38bdf8" stroke-width="2"/><polyline pathLength="1" points="${pt(a)}" fill="none" stroke="#a78bfa" stroke-width="2"/><text x="0" y="${H+13}">Sep 7</text><text x="${W-30}" y="${H+13}">Oct 7</text></svg><div class="muted">${B('Meetings','p')} ${B('Knowledge items','c')}</div>`}
const hbars=d=>d.map(([n,v,m])=>`<div style="margin:7px 0"><div class="row sp" style="margin:0"><span>${n}</span><b>${v}</b></div>${bar(v/m*100)}</div>`).join('');
const rnd=(n,m)=>Array.from({length:n},(_,i)=>Math.round(m*(.4+.35*Math.sin(i/3)+Math.random()*.25)));

const meetRow=m=>`<tr data-go="meeting" data-i="${MEET.indexOf(m)}"><td><b>${m[0]}</b></td><td>${m[1]}</td><td>${m[2]}</td><td>${m[3]}</td><td>${m[4]} min</td><td>${B(m[5],m[5]=='Standard'?'':'p')}</td><td>${bs(m[6])}</td><td>${m[7]} insights</td><td><button>Open</button></td></tr>`;
const MH=["Meeting","Project","Date","Participants","Duration","Privacy Mode","Processing Status","Knowledge","Actions"];

const P={
overview:()=>PH('Organizational Intelligence','Secure meeting intelligence, decision continuity and organizational memory.')+`<div class="grid g6">${kpi('Meetings',128,'+12%','▤')+kpi('Decisions',342,'+8%','◆')+kpi('Active Actions',76,'-4%','✓')+kpi('Open Risks',18,'+3','▲')+kpi('Privacy Events',27,'+9','⛨')+kpi('Pending Verification',12,'-2','◎')}</div>
<div class="grid g2"><div class="card"><h2>Processing Activity (30 days)</h2>${area(rnd(30,8),rnd(30,40))}</div>
<div class="card"><h2>Decision Continuity</h2>${hbars([['Decisions made',342,342],['Modified',58,342],['Potentially reversed',9,342],['Unresolved',21,342]])}</div></div>
<div class="grid g2"><div class="card"><h2>Privacy Firewall Activity</h2>${hbars([['Detected',212,300],['Redacted',74,300],['Masked',96,300],['Blocked',27,300],['Allowed',1860,1860]])}</div>
<div class="card"><h2>Knowledge Growth</h2>${hbars([['Entities',1284,1284],['Decisions',342,1284],['Actions',418,1284],['Risks',92,1284],['Questions',137,1284]])}</div></div>
<h2>Attention Required</h2><div class="grid g2"><div class="card warn"><b>⚠ Decision change</b><p>"Use PostgreSQL" may be reversed by "Evaluate MongoDB".</p><button data-go="decisions">Review</button></div>
<div class="card crit"><b>High-risk issue</b><p>Credentials spoken in Q4 Security Briefing.</p><button data-go="risks">Open risk</button></div>
<div class="card"><b>Verification required</b><p>12 extracted items await human confirmation.</p><button data-go="meetings">Verify</button></div>
<div class="card"><b>Retention event</b><p>Raw audio batch 118 deleted at 09:00 (24h policy).</p><button data-go="audit">View log</button></div></div>
<h2>Recent Meetings</h2>${TB(MH,MEET.slice(0,5).map(meetRow))}`,

meetings:()=>{const f=S.mf||'All',L=MEET.filter(m=>f=='All'||(f=='Recent'?m[2].includes('Oct'):m[6]==f||(f=='Needs Review'&&m[6]==f)));
return `<div class="row sp"><div class="row"><button class="pri" data-modal="newm">+ New Meeting</button><button data-toast="Import simulated">Import</button><button data-toast="Filter panel simulated">Filter</button><button data-toast="Export simulated">Export</button></div></div>${tabs('mf',['All','Recent','Processing','Completed','Needs Review','Archived'],f)}${TB(MH,L.map(meetRow))}`},

meeting:()=>{const m=MEET[S.meet],t=S.tab.mw||'Summary';const T={
Summary:`<div class="grid g2"><div class="card ai"><h2>✦ AI Summary</h2><p>The team confirmed PostgreSQL for Atlas, flagged a six-week migration estimate against the Nov 15 freeze, and assigned a migration plan to Rahul Kumar. A prototype of a document store was raised as an alternative.</p>${B('Confidence 92%','p')}</div>
<div class="card"><h2>Key Takeaways</h2><ul><li>PostgreSQL remains the primary database</li><li>Migration may miss release</li><li>MongoDB prototype requested</li></ul><h2>Open Questions</h2><ul><li>Can the freeze move by two weeks?</li></ul></div>
<div class="card"><h2>Topics</h2>${['Database','Migration','Release','Backend'].map(x=>B(x,'c')).join(' ')}<h2 style="margin-top:12px">Participants</h2>${m[3]} participants (names masked per ${m[5]} mode)</div></div>`,
Transcript:`<div class="tx"><p>[00:02] <span class="red">[PERSON]</span>: Let's start with the database question.</p><p>[00:09] Rahul: We decided on PostgreSQL in August.</p><p>[00:15] <span class="red">[API KEY BLOCKED]</span></p><p>[00:31] Dan: The freeze is <span class="red">[CONFIDENTIAL]</span>.</p></div><p class="muted">Raw transcript is deleted 72h after meeting.</p>`,
Decisions:TB(["Decision","Status","Owner","Date","Confidence","Related Meetings"],DEC.slice(0,3).map(d=>`<tr><td>${d[0]}</td><td>${bs(d[4])}</td><td>${d[2]}</td><td>${d[3]}</td><td>${d[5]}%</td><td>Architecture Review #1, Oct 7</td></tr>`))+`<div class="card warn">⚠ Later meeting (Oct 7) introduced a potential change: <b>Evaluate MongoDB</b></div><h2 style="margin-top:14px">Decision Evolution</h2><div class="card">Aug 14 · Proposed → Aug 14 · Active → Oct 7 · Under Review</div>`,
Actions:TB(["Task","Owner","Deadline","Review"],ACT.slice(0,3).map(a=>`<tr><td>${a[0]}</td><td>${a[1]}</td><td>${a[3]}</td><td><button data-toast="Confirmed">Confirm</button> <button data-toast="Edit simulated">Edit</button> <button data-toast="Rejected">Reject</button> <button data-toast="History simulated">History</button></td></tr>`)),
Risks:TB(["Risk","Severity","Status"],RISK.slice(0,2).map(r=>`<tr data-drawer="${RISK.indexOf(r)}"><td>${r[0]}</td><td>${bs(r[2])}</td><td>${r[5]}</td></tr>`)),
Entities:ENT.map(e=>B(e[0]+' · '+e[1],'c')).join(' '),
Timeline:`<div class="card">00:00 Intro · 12:40 Database debate · 31:10 Risk raised · 44:00 Action assigned · 54:00 End</div>`,
"Privacy Events":TB(["Item","Category","Action"],PEV.slice(0,3).map(p=>`<tr><td>${p[2]}</td><td>${p[3]}</td><td>${bs(p[5])}</td></tr>`)),
"Related Meetings":TB(["Meeting","Similarity"],KB.slice(0,3).map(k=>`<tr><td>${k[0]}</td><td>${k[2]}%</td></tr>`))};
return `<div class="card"><div class="row sp"><div><h2 style="font-size:18px">${m[0]}</h2>${m[1]} · ${m[2]} · ${m[3]} participants · ${B(m[5],'p')} ${bs(m[6])}</div><div class="row"><button data-go="meetings">Back</button><button data-ai="1">✦ Ask AI</button><button data-toast="Exported (simulated)">Export</button><button class="pri" data-toast="Verified">Verify</button><button data-toast="More…">⋯</button></div></div></div><br>${tabs('mw',Object.keys(T),t)}${T[t]}`},

live:()=>`<div class="row sp"><div><b>Product Architecture Review</b> ${B('<span class="pulse">●</span> LIVE','r')} ${B('Confidential','p')} <span id="tm" class="muted">00:00</span> · 8 participants</div><div class="row"><button id="pz">Pause</button><button class="dng" data-modal="stop">Stop Processing</button><button data-go="settings">Privacy Settings</button></div></div>
<div class="grid g2" style="grid-template-columns:2fr 1fr"><div class="card"><h2>Live Transcript</h2><div class="tx" id="ltx"></div></div><div><div class="card"><h2>Privacy Firewall</h2><div id="lp"></div></div><br><div class="card ai"><h2>✦ AI Extraction</h2><div id="lx"></div></div></div></div>`,

privacy:()=>`<div class="row">${B('FIREWALL ACTIVE','g')} ${B('Protection: HIGH','g')} ${B('AI Gateway: PROTECTED','g')}</div>`+PH('Privacy Firewall','Inspect and control sensitive information before AI processing or permanent storage.')+
`<div class="grid g6">${[['Sensitive',212],['Redacted',74],['Masked',96],['Anonymized',31],['Blocked',27],['Allowed',1860]].map(x=>`<div class="card kpi"><small>${x[0]}</small><b>${x[1]}</b></div>`).join('')}</div>
<div class="row">${['Regex','NER','Contextual Classification','Secret Detection'].map(x=>B('● '+x+' — active','g')).join(' ')}</div><h2>Recent Privacy Events</h2>
${TB(["Timestamp","Meeting","Detected Item","Category","Method","Action","Severity"],PEV.map(p=>`<tr><td>${p[0]}</td><td>${p[1]}</td><td>${p[2]}</td><td>${p[3]}</td><td>${p[4]}</td><td>${bs(p[5])}</td><td>${bs(p[6])}</td></tr>`))}
<h2>Rules</h2>${TB(["Information","Enabled","Action","Severity",""],RULES.map(r=>`<tr><td>${r[0]}</td><td><input type="checkbox" class="tog" checked data-toast="Rule updated (audit logged)"></td><td>${r[1]}</td><td>${bs(r[2])}</td><td><button data-toast="Rule editor simulated">Edit</button></td></tr>`))}`,

memory:()=>PH('Organizational Memory','Structured knowledge retained from meetings.')+`<div class="grid g6">${[['Decisions','342'],['Actions','418'],['Risks','92'],['Questions','137'],['Entities','1,284'],['Relationships','2,936']].map(x=>`<div class="card kpi"><small>${x[0]}</small><b>${x[1]}</b></div>`).join('')}</div>${tabs('mm',['Overview','Decisions','Actions','Risks','Questions','Entities','Relationships'],S.tab.mm||'Overview')}
<div class="row"><input type="text" id="gs" placeholder="Search graph…"><select id="gf"><option value="">All types</option><option>P</option><option>J</option><option>D</option><option>O</option><option>T</option></select><span class="muted">Drag to pan · scroll to zoom · click a node</span></div>
<div class="grid g2" style="grid-template-columns:2.2fr 1fr"><svg id="graph"></svg><div class="card" id="gd">Select a node to see details.</div></div>`,

decisions:()=>PH('Decisions','Track decisions across meetings.')+`<div class="card warn"><h2>⚠ Potential Decision Changes</h2><div class="grid g3"><div>Previous<br><b>Use PostgreSQL</b></div><div>Current<br><b>Evaluate MongoDB</b></div><div>${B('⚠ Potential Change Detected','a')}<br><button data-go="context">Review</button> <button data-toast="Comparing Aug 14 vs Oct 7 (simulated)">Compare Meetings</button></div></div></div><br>
<div class="row"><select><option>All projects</option></select><select><option>All departments</option></select><select><option>All statuses</option>${['Active','Proposed','Superseded','Reversed','Under Review'].map(x=>`<option>${x}</option>`)}</select><select><option>Any owner</option></select></div>
${TB(["Decision","Project","Owner","Date","Status","Confidence","Last Updated"],DEC.map(d=>`<tr><td><b>${d[0]}</b></td><td>${d[1]}</td><td>${d[2]}</td><td>${d[3]}</td><td>${bs(d[4])}</td><td>${d[5]}%</td><td>${d[6]}</td></tr>`))}`,

actions:()=>{const v=S.view,sts=['Not Started','In Progress','Blocked','Completed','Overdue'];
const body=v=='Kanban'?`<div class="kan">${sts.map(s=>`<div><b>${s}</b><br><br>${S.acts.filter(a=>a[5]==s).map(a=>`<div class="card"><b>${a[0]}</b><br><span class="muted">${a[1]} · ${a[3]}</span> ${bs(a[4])}</div>`).join('')}</div>`).join('')}</div>`
:v=='Timeline'?`<div class="card">${[...S.acts].sort((a,b)=>a[3].localeCompare(b[3])).map(a=>`<div style="padding:6px 0;border-bottom:1px solid var(--bd)"><b>${a[3]}</b> — ${a[0]} (${a[1]}) ${bs(a[5])}</div>`).join('')}</div>`
:TB(["Task","Owner","Project","Deadline","Priority","Status","Source Meeting",""],S.acts.map((a,i)=>`<tr><td>${a[0]}</td><td>${a[1]}</td><td>${a[2]}</td><td>${a[3]}</td><td>${bs(a[4])}</td><td>${bs(a[5])}</td><td>${a[6]}</td><td><button data-done="${i}">Complete</button> <button data-owner="${i}">Owner</button> <button data-toast="Deadline picker simulated">Deadline</button></td></tr>`));
return `<div class="row sp"><div class="tabs" style="margin:0" data-tabs="view">${['List','Kanban','Timeline'].map(x=>`<button class="${x==v?'on':''}" data-t="${x}">${x}</button>`).join('')}</div><button class="pri" data-add="1">+ Add Action</button></div>${body}`},

risks:()=>PH('Risks & Issues','Risks detected from meeting context.')+`<div class="grid g6">${[['Critical',1],['High',1],['Medium',1],['Low',1]].map(x=>`<div class="card kpi"><small>${x[0]}</small><b class="">${x[1]}</b>${bs(x[0])}</div>`).join('')}</div>${TB(["Risk","Project","Severity","Owner","Detected","Status","Source"],RISK.map((r,i)=>`<tr data-drawer="${i}"><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${bs(r[2])}</td><td>${r[3]}</td><td>${r[4]}</td><td>${bs(r[5])}</td><td>${r[6]}</td></tr>`))}`,

entities:()=>PH('Entities','People, projects, technology and topics resolved across meetings.')+`<div class="row">${['All','Person','Organization','Project','Location','Technology','Document','Topic'].map(x=>`<button data-ef="${x}" class="${(S.ef||'All')==x?'pri':''}">${x}</button>`).join('')}</div>
${TB(["Entity","Type","Mentions","Meetings","Confidence","Relationships"],ENT.filter(e=>(S.ef||'All')=='All'||e[1]==S.ef).map((e,i)=>`<tr data-drawer="e${ENT.indexOf(e)}"><td><b>${e[0]}</b></td><td>${e[1]}</td><td>${e[2]} mentions</td><td>${e[3]} meetings</td><td>${e[4]}%</td><td>${e[5]}</td></tr>`))}
<div class="card ai"><h2>Entity Resolution</h2>Rahul · R. Kumar · Rahul K. → <b>Rahul Kumar</b> ${B('Similarity 94%','p')}<br><br><button class="pri" data-modal="idm">Review Identity Match</button></div>`,

search:()=>PH('Semantic Search','Find knowledge by meaning, not exact keywords.')+`<input type="text" class="big" id="sq" placeholder="Ask about previous decisions, risks, actions or discussions..." value="What did we decide about the database architecture?"><div class="row" style="margin-top:10px"><select><option>Any date</option></select><select id="sp"><option value="">All projects</option><option>Project Atlas</option><option>Procurement</option></select><select><option>All people</option></select><select id="st"><option value="">All knowledge types</option><option>Decision</option><option>Risk</option><option>Action</option></select><select><option>All departments</option></select><button class="pri" id="sgo">Search</button></div><div id="sr"></div>`,

context:()=>PH('Context Explorer','See relevant organizational history retrieved for a new meeting.')+`<div class="grid g3"><div class="card"><h2>Current Query</h2><p>"We are considering changing the database architecture."</p><h2>Entities</h2>${['PostgreSQL','MongoDB','Backend Team','Project Atlas'].map(x=>B(x,'c')).join(' ')}</div>
<div class="card"><h2>Retrieval Pipeline</h2><div class="flow"><div>Entity Matching</div><span>↓</span><div>Semantic Retrieval</div><span>↓</span><div>Context Builder</div><span>↓</span><div>✦ AI Reasoning</div></div></div>
<div class="card"><h2>Retrieved</h2>${['Meeting #42','Meeting #38','Decision #17','Risk #12'].map(x=>`<div style="padding:5px 0">${x}</div>`).join('')}<div class="card warn" style="margin-top:10px"><b>⚠ Potential decision change detected</b><br>Confidence: <b>91%</b></div></div></div>`,

reports:()=>PH('Reports','Generate and export verified intelligence reports.')+`<div class="grid g3">${REPORTS.map(r=>`<div class="card"><b>${r[0]}</b><br><span class="muted">${r[1]} · ${r[2]}</span> ${bs(r[3])}<div style="margin:10px 0">${area(rnd(10,5),rnd(10,9)).split('<div')[0]}</div><button data-toast="Opening preview (simulated)">View</button> <button data-toast="Generating…">Generate</button> <button data-toast="Exported (simulated)">Export</button></div>`).join('')}</div>`,

audit:()=>PH('Audit Logs','Every access and policy change is recorded.')+`<div class="row"><select><option>All users</option></select><select><option>All actions</option></select><select><option>Today</option></select><select><option>All resources</option></select><select><option>All severities</option></select><button class="pri" data-toast="Audit log exported (simulated)">Export Audit Log</button></div>${TB(["Timestamp","User","Action","Resource","IP / Session","Result"],AUDIT.map(a=>`<tr>${a.slice(0,5).map(x=>`<td>${x}</td>`).join('')}<td>${bs(a[5]=='Blocked'?'Blocked':a[5])}</td></tr>`))}`,

retention:()=>`<div class="row">${B('Data Protection Healthy','g')}</div>`+PH('Retention & Privacy','Raw data is temporary. Verified knowledge is long-term.')+`<div class="grid g6" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">${[['RAW AUDIO','Temporary','Delete after 24h · Automatic deletion','a'],['RAW TRANSCRIPT','Temporary','Delete after 72h · Automatic deletion','a'],['STRUCTURED KNOWLEDGE','Long-term','Protected','g'],['EMBEDDINGS','Long-term','Protected','g']].map(x=>`<div class="card"><small class="muted">${x[0]}</small><br>${B(x[1],x[3])}<p>${x[2]}</p></div>`).join('')}</div>
<div class="card"><h2>Lifecycle</h2><div class="flow">${['Meeting','Temporary Audio','Transcript','Privacy Firewall','Structured Knowledge','Secure Memory','Automatic Raw Data Deletion'].map(x=>`<div>${x}</div>`).join('<span>↓</span>')}</div></div><br>
<div class="row"><button class="dng" data-modal="delraw">Delete raw data now</button><button class="dng" data-modal="delmeet">Delete a meeting's knowledge</button></div>`,

settings:()=>{const t=S.tab.st||'General',F={General:'Language: English · Timezone: IST',Organization:'Northfield Org · 214 members',AI:`Provider: <b>Gemini (Mock)</b><br>Embedding Model: <b>Mock Embedding Model</b><br>Context Retrieval: ${B('Enabled','g')}<br>Max Context: <b>10 records</b>`,Privacy:`Mode: <select><option>Standard</option><option selected>Confidential</option><option>Maximum Privacy</option></select><br><br>Retention — Audio: 24h · Transcript: 72h · Knowledge: Long-term`,Security:`2FA: ${B('On','g')} · Session timeout: 30 min · Audit logging: ${B('On','g')} · RBAC: ${B('Enforced','g')}`,Notifications:'Email digests: daily · Alerts: critical only',Integrations:'No external integrations (prototype).'};
return PH('Settings','Visual only — nothing is saved.')+tabs('st',Object.keys(F),t)+`<div class="card" style="line-height:2">${F[t]}</div><br><button class="pri" data-toast="Settings saved (simulated)">Save</button>`},
  // Placeholder renderers for pages provided by appModules.js (ES module, loads async).
  // These stubs are replaced at runtime when appModules.js patches P.upload / P.livetranslate.
  upload:()=>`<div class="card" style="border-color:var(--c)"><div class="row" style="margin:0"><span class="pulse" style="color:var(--c)">●</span><b>Loading Upload Meeting…</b></div><p class="muted">Initialising local processing modules. If this persists, check that your browser supports ES modules and WebAssembly.</p></div>`,
  livetranslate:()=>`<div class="card" style="border-color:var(--c)"><div class="row" style="margin:0"><span class="pulse" style="color:var(--c)">●</span><b>Loading Live Translation…</b></div><p class="muted">Initialising local Whisper and translation models.</p></div>`,
};

const TITLE=Object.fromEntries(NAV.map(n=>[n[0],n[1]]));TITLE.meeting='Meeting Workspace';
TITLE.upload='Upload Meeting';TITLE.livetranslate='Live Translation';

// Expose P, go, render on window so appModules.js (ES module) can patch them
window.P=P;window.TITLE=TITLE;


function go(){const h=(location.hash.slice(1)||'overview'),k=P[h]?h:'overview';clearInterval(timer);
$('#view').innerHTML=P[k]();enter();$('#ptitle').textContent=TITLE[k];$('#crumb').textContent='MEMORA / '+TITLE[k];
$$('#nav a').forEach(a=>a.classList.toggle('on',a.dataset.k==(k=='meeting'?'meetings':k)));$('#side').classList.remove('open');$('#main').scrollTop=0;
({live,memory:graph,search:()=>runSearch()})[k]?.();}
function render(){const h=location.hash.slice(1)||'overview';const y=$('#main').scrollTop;$('#view').innerHTML=P[P[h]?h:'overview']();fx();$('#main').scrollTop=y;({memory:graph,search:runSearch,live})[h]?.()}

function live(){const L=["Welcome everyone, let's review the database plan.","Rahul: our staging key is {K} for the cluster.","We decided to keep PostgreSQL for now.","Can {P} own the migration plan?","Risk: the migration might slip past the freeze.","The pricing strategy is {C}, keep that internal.","Action: Rahul drafts the plan by Oct 15.","Question: can the release freeze move?"];
let s=0,i=0,paused=false,c={s:12,r:7,m:3,b:2,a:31,d:3,ac:5,ri:2,q:4};const T=$('#ltx');
const upd=()=>{$('#lp').innerHTML=[['Sensitive',c.s],['Redacted',c.r],['Masked',c.m],['Blocked',c.b],['Allowed',c.a]].map(x=>`<div class="row sp" style="margin:3px 0"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('');$('#lx').innerHTML=[['Decisions',c.d],['Actions',c.ac],['Risks',c.ri],['Questions',c.q]].map(x=>`<div class="row sp" style="margin:3px 0"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('')};upd();
$('#pz').onclick=e=>{paused=!paused;e.target.textContent=paused?'Resume':'Pause'};
timer=setInterval(()=>{if(paused)return;s++;$('#tm').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');if(s%2)return;
let l=L[i++%L.length].replace('{K}','<span class="red">[API KEY BLOCKED]</span>').replace('{P}','<span class="red">[PERSON]</span>').replace('{C}','<span class="red">[CONFIDENTIAL]</span>');
if(l.includes('BLOCKED'))c.b++;else if(l.includes('PERSON'))c.m++;else if(l.includes('CONFIDENTIAL'))c.r++;else c.a++;if(/red/.test(l))c.s++;
if(/decided/.test(l))c.d++;if(/Action/.test(l))c.ac++;if(/Risk/.test(l))c.ri++;if(/Question/.test(l))c.q++;
T.insertAdjacentHTML('beforeend',`<p>[${$('#tm').textContent}] ${l}</p>`);T.scrollTop=T.scrollHeight;upd()},1000)}

function graph(){const svg=$('#graph'),W=700,H=440,col={P:'#38bdf8',J:'#34d399',D:'#a78bfa',O:'#fbbf24',T:'#8a99b5'};let vb=[0,0,W,H],sel=-1;
const pos=GN.map((_,i)=>[W/2+210*Math.cos(i*6.283/GN.length),H/2+160*Math.sin(i*6.283/GN.length)]);
const draw=()=>{const q=($('#gs')?.value||'').toLowerCase(),f=$('#gf')?.value;svg.setAttribute('viewBox',vb.join(' '));
svg.innerHTML=GE.map(e=>`<line x1="${pos[e[0]][0]}" y1="${pos[e[0]][1]}" x2="${pos[e[1]][0]}" y2="${pos[e[1]][1]}" style="stroke:var(--bd)"/><text x="${(pos[e[0]][0]+pos[e[1]][0])/2}" y="${(pos[e[0]][1]+pos[e[1]][1])/2}">${e[2]}</text>`).join('')+
GN.map((n,i)=>{const dim=(f&&n[1]!=f)||(q&&!n[0].toLowerCase().includes(q));return `<g data-n="${i}" opacity="${dim?.2:1}" style="cursor:pointer"><circle cx="${pos[i][0]}" cy="${pos[i][1]}" r="${i==sel?16:11}" fill="${col[n[1]]}" stroke="${i==sel?'#fff':'none'}"/><text x="${pos[i][0]+14}" y="${pos[i][1]+4}" style="fill:var(--t)">${n[0]}</text></g>`}).join('')};draw();
svg.onwheel=e=>{e.preventDefault();const k=e.deltaY>0?1.1:.9;vb=[vb[0]+vb[2]*(1-k)/2,vb[1]+vb[3]*(1-k)/2,vb[2]*k,vb[3]*k];draw()};
let d=null;svg.onmousedown=e=>d=[e.clientX,e.clientY];window.onmouseup=()=>d=null;svg.onmousemove=e=>{if(!d)return;const k=vb[2]/svg.clientWidth;vb[0]-=(e.clientX-d[0])*k;vb[1]-=(e.clientY-d[1])*k;d=[e.clientX,e.clientY];draw()};
svg.onclick=e=>{const g=e.target.closest('[data-n]');if(!g)return;sel=+g.dataset.n;const n=GN[sel],rel=GE.filter(x=>x[0]==sel||x[1]==sel);draw();$('#gd').innerHTML=`<h2>${n[0]}</h2>${rel.map(x=>`<div>${GN[x[0]][0]} <span class="muted">${x[2]}</span> ${GN[x[1]][0]}</div>`).join('')}`};
$('#gs').oninput=draw;$('#gf').onchange=draw}

function runSearch(){const q=($('#sq')?.value||'').toLowerCase().split(/\W+/).filter(w=>w.length>3&&!['what','about','with'].includes(w));const p=$('#sp').value,t=$('#st').value;
const r=KB.filter(k=>(!p||k[5]==p)&&(!t||k[4]==t)).map(k=>[k,Math.min(99,k[2]+(q.some(w=>k[3].toLowerCase().includes(w))?3:-10))]).sort((a,b)=>b[1]-a[1]);
$('#sr').innerHTML=r.length?r.map(([k,s])=>`<div class="card" style="margin-bottom:10px"><div class="row sp" style="margin:0"><b>${k[0]}</b><span class="muted">${k[1]} · ${B(k[4],'c')}</span></div><p>“${k[3]}”</p><div class="row" style="margin:0"><span>${s}%</span><div class="bar" style="flex:1">${'<i style="width:'+s+'%"></i>'}</div></div></div>`).join(''):'<p class="muted">No matching knowledge. Try removing filters.</p>'}

function modal(h,dr){const o=$('#overlay');o.className=dr?'dr':'';o.innerHTML=`<div class="modal">${h}<br><br><button data-close="1">Close</button></div>`;o.hidden=false}
const confirmM=(t,msg)=>modal(`<h2>${t}</h2><p>${msg}</p><p class="muted">This is a simulated action.</p><button class="dng" data-close="1" data-toast="${t} — completed (simulated)">Confirm</button> `);
function drawer(i){if(String(i)[0]=='e'){const e=ENT[+i.slice(1)];return modal(`<h2>${e[0]}</h2>${B(e[1],'c')} ${e[4]}% confidence<p><b>Aliases:</b> ${e[0].split(' ')[0]}, ${e[0][0]}. ${e[0].split(' ').pop()}</p><p>${e[2]} mentions · ${e[3]} meetings · ${e[5]} relationships</p><p><b>Decisions:</b> Use PostgreSQL<br><b>Actions:</b> Draft database migration plan</p>`,1)}
const r=RISK[i];modal(`<h2>${r[0]}</h2>${bs(r[2])} ${bs(r[5])}<h2 style="margin-top:14px">Why detected</h2><p>${r[7]}</p><h2>Supporting excerpts</h2><p class="muted">“…six weeks… before the freeze on the fifteenth.”</p><h2>Related</h2><p><b>Decision:</b> Freeze release branch on Nov 15<br><b>Action:</b> Draft database migration plan<br><b>Entities:</b> Project Atlas, PostgreSQL</p>`,1)}

function palette(){modal(`<input type="text" class="big" id="pq" placeholder="Search meetings, decisions, people, projects, actions, risks, documents…" autofocus><div id="pr" style="margin-top:10px"></div>`);const f=()=>{const q=$('#pq').value.toLowerCase();const all=[...MEET.map(m=>['Meeting',m[0],'meetings']),...DEC.map(d=>['Decision',d[0],'decisions']),...ENT.map(e=>[e[1],e[0],'entities']),...ACT.map(a=>['Action',a[0],'actions']),...RISK.map(r=>['Risk',r[0],'risks'])].filter(x=>x[1].toLowerCase().includes(q)).slice(0,8);
$('#pr').innerHTML=all.map(x=>`<div style="padding:6px 0"><a href="#${x[2]}" data-close="1">${x[1]}</a> ${B(x[0],'c')}</div>`).join('')};f();$('#pq').oninput=f;$('#pq').focus()}

function aiPanel(){const a=$('#ai');a.hidden=!a.hidden;if(a.innerHTML)return;a.innerHTML=`<div class="row sp" style="margin:0"><div><b>Organizational AI</b><div class="muted">Ask questions about your organization's verified knowledge.</div></div><button data-ai="1">✕</button></div><div id="msgs"></div><div class="chips">${[["What decisions were made about Project Atlas?","decisions"],["Which decisions changed recently?","changed"],["What are our biggest unresolved risks?","risks"],["Who owns the database migration?","owns"],["What was discussed about PostgreSQL last month?","postgres"]].map(x=>`<button data-q="${x[1]}" data-t="${x[0]}">${x[0]}</button>`).join('')}</div><div class="row" style="margin:8px 0 0"><input type="text" id="aq" style="flex:1" placeholder="Ask…"><button class="pri" id="ask">Ask</button></div>`}
function ask(k,t){const m=$('#msgs');const ans=AIA[k]||AIA[Object.keys(AIA).find(x=>t.toLowerCase().includes(x))]||'Mock response: I found 3 related records in verified memory.';m.insertAdjacentHTML('beforeend',`<div class="msg u">${t}</div><div class="msg ai">${ans}<div class="muted" style="margin-top:6px">Sources: <a href="#meetings">Architecture Review</a> · <a href="#decisions">Decision #17</a> · <a href="#meetings">Migration Planning</a></div><div class="row" style="margin:6px 0 0"><button data-toast="Copied">Copy</button><button data-go="meetings">Open Source</button></div></div>`);m.scrollTop=m.scrollHeight}

document.addEventListener('click',e=>{const t=e.target,c=s=>t.closest(s);let x;
if(x=c('[data-close]')){if(!c('.modal')||t.dataset.close||x)$('#overlay').hidden=true}else if(t.id=='overlay')t.hidden=true;
if(x=c('[data-toast]')&&t.type!='checkbox'?c('[data-toast]'):0)toast(x.dataset.toast);
if(x=c('[data-go]')){if(x.dataset.i!=null)S.meet=+x.dataset.i;if(location.hash=='#'+x.dataset.go)go();else location.hash=x.dataset.go}
if(x=c('[data-drawer]'))drawer(x.dataset.drawer);
if(x=c('[data-tabs] button')){const id=x.parentNode.dataset.tabs;if(id=='view')S.view=x.dataset.t;else if(id=='mf')S.mf=x.dataset.t;else S.tab[id]=x.dataset.t;render()}
if(x=c('[data-ef]')){S.ef=x.dataset.ef;render()}
if(x=c('[data-done]')){S.acts[+x.dataset.done][5]='Completed';render();toast('Action completed')}
if(x=c('[data-owner]')){const a=S.acts[+x.dataset.owner],o=['Rahul Kumar','Priya Nair','Meera Iyer','Asha Verma'];a[1]=o[(o.indexOf(a[1])+1)%4];render();toast('Owner changed to '+a[1])}
if(c('[data-add]')){S.acts.unshift(['New action item','Asha Verma','Project Atlas','Oct 20','Medium','Not Started','Manual']);render();toast('Action added')}
if(x=c('[data-modal]')){const k=x.dataset.modal;if(k=='newm')modal('<h2>New Meeting</h2><p><input type="text" placeholder="Title" style="width:100%"></p><p><select><option>Standard</option><option>Confidential</option><option>Maximum Privacy</option></select></p><button class="pri" data-close="1" data-go="live">Start processing</button>');
else if(k=='stop')confirmM('Stop Processing','Stop live processing? Raw audio will be queued for deletion.');
else if(k=='delraw')confirmM('Delete raw data','Permanently delete all raw audio and transcripts now?');
else if(k=='delmeet')confirmM("Delete meeting knowledge",'Permanently delete structured knowledge for this meeting?');
else if(k=='idm')modal('<h2>Identity Match</h2><p>Rahul · R. Kumar · Rahul K. appear to be the same person (94%).</p><button class="pri" data-close="1" data-toast="Identities merged (simulated)">Merge</button> <button data-close="1">Keep separate</button>')}
if(c('[data-ai]'))aiPanel();if(x=c('[data-q]'))ask(x.dataset.q,x.dataset.t);
if(t.id=='ask'&&$('#aq').value){ask('',$('#aq').value);$('#aq').value=''}if(t.id=='sgo')runSearch();
if(c('#gsearch'))palette();if(t.id=='aibtn')aiPanel();if(t.id=='theme')setTheme();if(t.id=='collapse')$('#side').classList.toggle('min');if(t.id=='burger')$('#side').classList.toggle('open');
if(t.id=='logout'){e.preventDefault();toast('Logged out (simulated)')}if(t.id=='notif')toast('3 new notifications (simulated)');if(t.id=='help')toast('Help center (simulated)')});
document.addEventListener('change',e=>{if(e.target.dataset.toast)toast(e.target.dataset.toast)});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key=='k'){e.preventDefault();palette()}if(e.key=='Escape')$('#overlay').hidden=true;if(e.key=='Enter'&&e.target.id=='sq')runSearch();if(e.key=='Enter'&&e.target.id=='aq')$('#ask').click()});
$('#nav').innerHTML=NAV.map(n=>`<a href="#${n[0]}" data-k="${n[0]}"><span>${n[2]}</span><span class="lbl">${n[1]}</span></a>`).join('');
window.addEventListener('hashchange',go);go();

function fx(){$$('.kpi b').forEach(e=>{const raw=e.textContent,n=+raw.replace(/,/g,'');if(isNaN(n)||matchMedia('(prefers-reduced-motion:reduce)').matches)return;const t0=performance.now();(function f(t){const p=Math.min(1,(t-t0)/700),v=Math.round(n*(1-Math.pow(1-p,3)));e.textContent=raw.includes(',')?v.toLocaleString('en-US'):v;if(p<1)requestAnimationFrame(f)})(t0)})}
function enter(){const v=$('#view');v.classList.remove('enter');void v.offsetWidth;v.classList.add('enter');fx()}
function setTheme(){const d=document.documentElement,n=d.dataset.theme=='light'?'dark':'light';d.dataset.theme=n;try{localStorage.memTheme=n}catch(e){}icon()}
function icon(){$('#theme').textContent=document.documentElement.dataset.theme=='light'?'☾':'☀'}icon();
