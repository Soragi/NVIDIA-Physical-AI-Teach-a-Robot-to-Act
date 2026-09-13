'use strict';
const $=id=>document.getElementById(id),labels=['Test','Demonstrations','Train','Retest','Compare'];
let state={},stage=0,csrf='',pending=false,lastPhase='',lastFrame=0,localError='',connectionError='',lastOperation='';
function text(id,value){$(id).textContent=value??'—'}
function show(n){stage=Math.max(0,Math.min(4,n));for(let i=0;i<5;i++){$('stage'+i).hidden=i!==stage;$('nav'+i).classList.toggle('active',i===stage);$('nav'+i).setAttribute('aria-current',i===stage?'step':'false')}text('stepLabel',`${stage+1} / 5`);$('back').disabled=stage===0;$('next').hidden=stage===4;renderLive()}
labels.forEach((label,i)=>{const b=document.createElement('button');b.id='nav'+i;b.textContent=`0${i+1} / ${label}`;b.onclick=()=>show(i);$('steps').append(b)});
$('back').onclick=()=>show(stage-1);$('next').onclick=()=>show(stage+1);
async function post(route,data={}){if(pending)return;pending=true;localError='';render();try{const r=await fetch('/mission-api/training/'+route,{method:'POST',headers:{'Content-Type':'application/json','X-Mission-Token':csrf},body:JSON.stringify(data)});const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');await poll();return d}catch(e){localError=e.message;throw e}finally{pending=false;render()}}
$('before').onclick=async()=>{try{const raw=$('seed').value;await post('create',{seed:raw===''?null:Number(raw)});await post('before');show(0)}catch{}};
$('train').onclick=async()=>{try{await post('train',{steps:Number($('trainingSteps').value)})}catch{}};
$('after').onclick=async()=>{try{await post('after')}catch{}};
$('stop').onclick=async()=>{try{await post('stop')}catch{}};
$('restart').onclick=()=>{if(!state.busy){$('seed').value='';show(0)}};
$('playBoth').onclick=()=>{for(const id of ['compareBefore','compareAfter']){$(id).currentTime=0;$(id).play().catch(()=>{})}};
$('download').onclick=()=>{const {csrf:ignored,...report}=state;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`physical-ai-training-${state.session_id||'report'}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
function video(id,result){const node=$(id);if(!result){node.hidden=true;node.removeAttribute('src');return}const src='/mission-api/training/media/'+result.run_id+'/episode.mp4';if(node.getAttribute('src')!==src)node.src=src;node.hidden=false}
function renderLive(){const visible=state.busy&&!connectionError&&state.phase!=='reviewing_evidence'&&['before','after'].includes(state.operation)&&stage===(state.operation==='before'?0:3);$('live').hidden=!visible;if(!visible)return;const p=state.progress||{};text('livePhase',p.phase||'LOADING');text('control',`${p.step||0} / 500`);if(p.run_id&&p.step>0&&Date.now()-lastFrame>900){lastFrame=Date.now();$('liveFrame').src=`/mission-api/training/media/${p.run_id}/frame.jpg?t=${lastFrame}`;$('liveFrame').hidden=false;$('loadingCamera').hidden=true}else if(!p.run_id||!p.step){$('liveFrame').hidden=true;$('loadingCamera').hidden=false}}
function render(){const busy=state.busy||pending;const t=state.training||{};const messages={ready:'Experiment ready. Run the early checkpoint.',preparing:'Preparing GPU resources…',evaluating:'Running a fresh simulation and collecting evidence…',training:'Training is changing model weights. Follow the measured loss below.',trained:'Your checkpoint is saved. Retest it in the same scene.',evaluated:'Evaluation recorded. Continue to the next step.',complete:'Comparison saved. Inspect the measured outcome.',stopping:'Stopping safely and saving available evidence…',stopped:'Operation stopped. No success inferred; create a new experiment.',interrupted:'Operation was interrupted. Create a new experiment.',error:'An operation failed. Inspect the error before continuing.'};text('status',messages[state.phase]||'Create your first experiment.');$('stop').disabled=!state.busy||pending;$('before').disabled=busy;$('train').disabled=busy||!state.before||!!state.training;$('after').disabled=busy||!state.after_checkpoint||!!state.after;$('restart').disabled=busy;text('seedReadout',state.scenario?`Recorded scene seed: ${state.scenario.seed} · Experiment: ${state.session_id}`:'');
const error=localError||connectionError||state.error;text('error',error||'');$('error').hidden=!error;
if(connectionError||!csrf)for(const id of ['before','train','after'])$(id).disabled=true;
$('stop').hidden=!(state.busy||state.restore_nemotron);$('stop').disabled=pending;text('stop',state.busy?'Stop operation':'Release training GPU');$('resourceHint').hidden=!state.restore_nemotron;text('resourceHint','GPU 0 is reserved for this experiment. Nemotron returns after comparison, stop, or an explicit GPU release.');
if(state.phase==='releasing')text('status','Releasing the training GPU and restoring Nemotron…');
if(state.phase==='preparing'&&state.gpu_status)text('status',state.gpu_status);
if(!state.busy&&!state.before)text('seedReadout','Your scene seed is recorded when the test starts.');
if(state.phase==='reviewing_evidence')text('status','Robot execution is recorded. Cosmos is reviewing the video and VSS is indexing the evidence…');
video('beforeVideo',state.before);video('afterVideo',state.after);
video('compareBefore',state.before);video('compareAfter',state.after);$('playBoth').disabled=!state.before||!state.after;text('diagnosis',state.before?.cosmos?.answer||state.before?.evidence_error||'Run the early checkpoint to collect a visual assessment.');
for(const side of ['before','after']){const r=state[side];text(side+'Result',r?`Run ${r.run_id} · ${r.success?'Task completed':'Task not completed'} · ${r.episode_steps} control steps · ${r.inference_calls} inference calls`:'')}
text('optimizer',t.max_steps?`${t.training_steps} / ${t.max_steps}`:'—');const losses=t.loss_history||[];text('loss',losses.length?losses.at(-1).loss:'—');text('elapsed',t.elapsed_s?`${Math.round(t.elapsed_s)} s`:'—');text('weights',t.weights_changed?'Verified: trainable weights changed. The retest loads this newly produced checkpoint.':t.phase?'Training is in progress; checkpoint verification is pending.':'No checkpoint has been trained in this experiment yet.');text('trainingEvidence',JSON.stringify(t,null,2));
drawLoss(losses,t.max_steps);
const b=state.before,a=state.after,c=state.comparison;text('verdict',c?(c.improved?'Measured improvement: the early policy failed and your trained policy completed this scene.':`Early checkpoint: ${c.before_success?'success':'failure'}. Trained checkpoint: ${c.after_success?'success':'failure'}. This experiment ${c.before_success===c.after_success?'did not change the measured outcome':'regressed'}.`):'Complete both evaluations to measure the difference.');$('comparison').replaceChildren();if(b||a){for(const [label,key] of [['Task completed','success'],['Control steps','episode_steps'],['Inference calls','inference_calls'],['Wall time (seconds)','wall_time_s'],['Run ID','run_id']]){const row=document.createElement('tr');for(const value of [label,b?.[key],a?.[key]]){const cell=document.createElement('td');cell.textContent=value??'Pending';row.append(cell)}$('comparison').append(row)}}text('evidence',JSON.stringify({before:b&&{cosmos:b.cosmos,vss:b.vss,error:b.evidence_error},after:a&&{cosmos:a.cosmos,vss:a.vss,error:a.evidence_error}},null,2));const {csrf:ignored,...report}=state;text('report',JSON.stringify(report,null,2));renderLive()}
function drawLoss(losses,total){if(losses.length<2){$('lossChart').innerHTML='<text x="20" y="90">Loss appears after training starts.</text>';return}const max=Math.max(...losses.map(x=>x.loss)),min=Math.min(...losses.map(x=>x.loss));$('lossChart').replaceChildren();function svg(tag,attrs,value){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))node.setAttribute(k,v);if(value!==undefined)node.textContent=value;$('lossChart').append(node)}svg('polyline',{points:losses.map(x=>`${75+695*x.step/total},${140-120*(x.loss-min)/(max-min||1)}`).join(' '),fill:'none',stroke:'#76b900','stroke-width':2});for(const [x,y,label]of [[5,24,max.toPrecision(3)],[5,140,min.toPrecision(3)],[75,170,'0'],[340,170,'Optimizer steps'],[725,170,String(total)]])svg('text',{x,y,style:'font-size:12px'},label)}
async function poll(){
  try{
    const r=await fetch('/mission-api/training/state',{cache:'no-store'});
    if(!r.ok)throw Error('Workshop API unavailable');
    state=await r.json();csrf=state.csrf;connectionError='';
    text('connection',state.busy?'● OPERATION ACTIVE':'● CONNECTED');
    const operation=`${state.session_id}:${state.operation}`;
    if(state.busy&&operation!==lastOperation){const target={before:0,train:2,after:3}[state.operation];if(target!==undefined)show(target)}
    lastOperation=operation;
    if(state.phase!==lastPhase){if(state.phase==='trained')show(3);if(state.phase==='complete')show(4);lastPhase=state.phase}
    render();
  }catch(e){text('connection','CONNECTION LOST');connectionError=e.message;render()}
}
show(0);poll();setInterval(poll,1200);
