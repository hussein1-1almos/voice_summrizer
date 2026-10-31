const $ = id => document.getElementById(id);
const state = { file:null, ws:null, stream:null, audioContext:null, processor:null, source:null, timer:null, startedAt:0, elapsed:0, transcript:"", summary:"", sampleRate:48000, pcmQueue:[], activeMode:null, displayStream:null };
const API = location.origin;
const WS_BASE = (location.protocol === "https:" ? "wss://" : "ws://") + location.host;

async function checkServer(){
  try { const r=await fetch("/health"); if(!r.ok) throw Error(); $("serverState").className="status ok"; $("serverState").innerHTML="<i></i> Server connected"; }
  catch { $("serverState").className="status bad"; $("serverState").innerHTML="<i></i> Server status unavailable"; }
}
checkServer();
document.querySelectorAll(".tab").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===b));
  document.querySelectorAll(".tab-panel").forEach(x=>x.classList.toggle("active",x.id==="panel-"+b.dataset.tab));
}));
function setState(s){$("sessionState").textContent=s;}
function updateTranscript(t){state.transcript=t; $("transcript").textContent=t||"No speech recognized yet."; $("transcriptCount").textContent=(t.trim()?t.trim().split(/\s+/).length:0)+" words"; $("downloadTxt").disabled=!t.trim();}
function updateSummary(t){state.summary=t; $("summary").textContent=t||"Waiting for enough speech to prepare a summary.";}
function resetResults(){updateTranscript("");updateSummary("");setState("Listening");}
function exportText(){const body="TRANSCRIPT\n\n"+state.transcript+"\n\nROLLING SUMMARY\n\n"+state.summary;const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([body],{type:"text/plain;charset=utf-8"}));a.download="voice_intelligence_notes.txt";a.click();URL.revokeObjectURL(a.href);}
$("downloadTxt").onclick=exportText;

const fileInput=$("audioFile");
fileInput.onchange=()=>selectFile(fileInput.files[0]);
function selectFile(f){if(!f)return;state.file=f;$("fileName").textContent=f.name;$("fileSize").textContent=(f.size/1048576).toFixed(2)+" MB";$("fileRow").hidden=false;$("uploadBtn").disabled=false;}
$("clearFile").onclick=()=>{state.file=null;fileInput.value="";$("fileRow").hidden=true;$("uploadBtn").disabled=true;};
const dz=$("dropzone");
dz.ondragover=e=>{e.preventDefault();dz.style.borderColor="#315eea";};
dz.ondragleave=()=>dz.style.borderColor="";
dz.ondrop=e=>{e.preventDefault();dz.style.borderColor="";selectFile(e.dataTransfer.files[0]);};
$("uploadBtn").onclick=async()=>{
 if(!state.file)return; $("uploadBtn").disabled=true;$("uploadProgress").hidden=false;$("uploadStatus").textContent="Uploading audio and waiting for transcription…";setState("Processing file");
 const fd=new FormData();fd.append("file",state.file);fd.append("language",$("uploadLanguage").value);fd.append("include_summary","true");
 try{
  const r=await fetch("/transcribe",{method:"POST",body:fd});const data=await r.json();if(!r.ok)throw Error(data.detail||JSON.stringify(data));
  updateTranscript(data.text||"");updateSummary(data.summary||"No summary returned."+(data.summary_error?"\n\n"+data.summary_error:""));setState("Completed");$("uploadStatus").textContent="Processing complete.";
 }catch(e){setState("Request failed");$("uploadStatus").textContent=e.message;}
 finally{$("uploadBtn").disabled=false;}
};

function floatTo16BitPCM(float32){const out=new Int16Array(float32.length);for(let i=0;i<float32.length;i++){let s=Math.max(-1,Math.min(1,float32[i]));out[i]=s<0?s*32768:s*32767;}return out.buffer;}
function startClock(timerId){state.startedAt=Date.now()-state.elapsed*1000;clearInterval(state.timer);state.timer=setInterval(()=>{state.elapsed=Math.floor((Date.now()-state.startedAt)/1000);$(timerId).textContent=fmt(state.elapsed);},500);}
function fmt(s){return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");}
async function startAudio(mode,language,stream,timerId){
 resetResults();state.stream=stream;state.activeMode=mode;
 const AudioCtx=window.AudioContext||window.webkitAudioContext;state.audioContext=new AudioCtx();await state.audioContext.resume();state.sampleRate=state.audioContext.sampleRate;
 state.source=state.audioContext.createMediaStreamSource(stream);state.processor=state.audioContext.createScriptProcessor(4096,1,1);
 state.pcmQueue=[];let queuedSamples=0;state.flushPending=()=>{if(!state.pcmQueue.length||!state.ws||state.ws.readyState!==WebSocket.OPEN)return;const n=state.pcmQueue.reduce((a,x)=>a+x.length,0);const joined=new Float32Array(n);let off=0;for(const part of state.pcmQueue){joined.set(part,off);off+=part.length;}state.pcmQueue=[];state.ws.send(floatTo16BitPCM(joined));};const SEND_SECONDS=5;const target=state.sampleRate*SEND_SECONDS;
 state.ws=new WebSocket(WS_BASE+"/ws/live");state.ws.binaryType="arraybuffer";
 state.ws.onopen=()=>{state.ws.send(JSON.stringify({type:"start",sample_rate:state.sampleRate,language,source:mode}));setState("Connected · listening");};
 state.ws.onmessage=e=>{let m;try{m=JSON.parse(e.data)}catch{return}
  if(m.type==="transcript"){const prefix=state.transcript?state.transcript+"\n":"";updateTranscript(prefix+m.text);setState("Transcribing");}
  if(m.type==="summary"){updateSummary(m.summary);setState("Summary updated");}
  if(m.type==="status")setState(m.message);
  if(m.type==="error")setState("Error: "+m.message);
  if(m.type==="finished"){updateTranscript(m.transcript||state.transcript);updateSummary(m.summary||state.summary);setState("Session finished");}
 };
 state.ws.onerror=()=>setState("WebSocket error — check server and network.");
 state.ws.onclose=()=>{if(state.activeMode)setState("Connection closed");};
 state.processor.onaudioprocess=e=>{
   if(!state.ws||state.ws.readyState!==WebSocket.OPEN)return;
   const ch=e.inputBuffer.getChannelData(0);const copy=new Float32Array(ch.length);copy.set(ch);state.pcmQueue.push(copy);queuedSamples+=copy.length;
   if(queuedSamples>=target){const joined=new Float32Array(queuedSamples);let off=0;for(const part of state.pcmQueue){joined.set(part,off);off+=part.length;}state.pcmQueue=[];queuedSamples=0;state.ws.send(floatTo16BitPCM(joined));}
 };
 state.source.connect(state.processor);state.processor.connect(state.audioContext.destination);
 // Mute monitor output while retaining ScriptProcessor processing.
 const mute=state.audioContext.createGain();mute.gain.value=0;state.processor.disconnect();state.processor.connect(mute);mute.connect(state.audioContext.destination);state.mute=mute;
 state.elapsed=0;startClock(timerId);
}
async function stopAudio(){
 clearInterval(state.timer);state.timer=null;
 try{
  if(state.ws&&state.ws.readyState===WebSocket.OPEN){if(state.flushPending)state.flushPending();state.ws.send(JSON.stringify({type:"stop"}));}
  if(state.processor){state.processor.onaudioprocess=null;state.processor.disconnect();}
  if(state.source)state.source.disconnect();
  if(state.mute)state.mute.disconnect();
  if(state.stream)state.stream.getTracks().forEach(t=>t.stop());
  if(state.displayStream&&state.displayStream!==state.stream)state.displayStream.getTracks().forEach(t=>t.stop());
  if(state.audioContext)await state.audioContext.close();
 }catch(e){console.warn(e)}
 state.stream=null;state.displayStream=null;state.activeMode=null;
}
$("micStart").onclick=async()=>{
 try{const s=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});$("micTitle").textContent="Microphone is active";$("micHint").textContent="Audio is being sent to your transcription server.";document.getElementById("panel-mic").classList.add("active-recording");$("micStart").disabled=true;$("micStop").disabled=false;await startAudio("microphone",$("micLanguage").value,s,"micTimer");}
 catch(e){setState("Microphone permission failed: "+e.message);}
};
$("micStop").onclick=async()=>{$("micStop").disabled=true;await stopAudio();$("micStart").disabled=false;$("micTitle").textContent="Microphone stopped";$("micHint").textContent="You can start another recording.";document.getElementById("panel-mic").classList.remove("active-recording");};

$("meetingStart").onclick=async()=>{
 try{
  const s=await navigator.mediaDevices.getDisplayMedia({video:true,audio:true});
  const audioTracks=s.getAudioTracks();if(!audioTracks.length){s.getTracks().forEach(t=>t.stop());throw Error("No tab audio was shared. Reopen sharing and enable “Share tab audio”.");}
  // Keep the selected display capture alive, but send only its audio track to the audio pipeline.
  state.displayStream=s;const audioOnly=new MediaStream(audioTracks);$("meetingTitle").textContent="Meeting tab audio is active";$("meetingHint").textContent="Shared tab audio is being processed."; $("meetingStart").disabled=true;$("meetingStop").disabled=false;
  await startAudio("meeting-tab",$("meetingLanguage").value,audioOnly,"meetingTimer");
  s.getVideoTracks().forEach(t=>t.addEventListener("ended",()=>stopMeeting()));
  s.getAudioTracks().forEach(t=>t.addEventListener("ended",()=>stopMeeting()));
 }catch(e){setState(e.message);}
};
async function stopMeeting(){ $("meetingStop").disabled=true;await stopAudio();$("meetingStart").disabled=false;$("meetingTitle").textContent="Meeting capture stopped";$("meetingHint").textContent="Share the meeting tab again to resume."; }
$("meetingStop").onclick=stopMeeting;
window.addEventListener("beforeunload",()=>{try{if(state.ws)state.ws.close();if(state.stream)state.stream.getTracks().forEach(t=>t.stop());}catch{}});
