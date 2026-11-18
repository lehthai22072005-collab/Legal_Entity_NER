"""Standalone entity training and portable checkpoint restoration."""
import collections,hashlib,json,math,os,random,time
from pathlib import Path
import torch
from .model import EntityExtractor
from .dataset import read_documents,dataset_sha256,write_json
from .entity_task import evaluate_entities,entity_policy

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def objective(metric):
    e=metric['entity_exact_span'];return (e['micro']['f1']+(e['macro_f1_supported_labels'] or 0))/2
def evaluate_model(model,documents,thresholds,topk):
    predictions=[model.infer(d['full_text'],max_boundaries=topk,**thresholds) for d in documents]
    return evaluate_entities(documents,predictions,model.ontology),predictions
def save_checkpoint(folder,model,extra):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);tmp=folder/'model.pt.tmp'
    torch.save({'state_dict':model.state_dict(),'config':model.config,'ontology':model.ontology,'extra':extra},tmp);os.replace(tmp,folder/'model.pt')
    write_json(folder/'config.json',model.config);write_json(folder/'training_state.json',extra);write_json(folder/'decoder_policy.json',model.config['decoder_policy'])
def load_checkpoint(folder,device='cpu'):
    folder=Path(folder);data=torch.load(folder/'model.pt',map_location='cpu',mmap=True,weights_only=False)
    config=dict(data['config']);assert config['entity_only'] and 'relations' not in data['ontology']
    if device=='cpu':torch.set_num_threads(min(2,torch.get_num_threads()))
    config.update(pretrained_path=str(Path(__file__).resolve().parents[1]/'resources/phobert-base'),initialize_pretrained=False)
    if config['encoder']=='phobert':
        from transformers.models.roberta.modeling_roberta import RobertaModel
    with torch.device('meta'):model=EntityExtractor(config,data['ontology'])
    model.load_state_dict(data['state_dict'],assign=True)
    for module in model.modules():
        for name,value in list(module.named_buffers(recurse=False)):
            if not value.is_meta:continue
            if name=='position_ids':restored=torch.arange(value.shape[-1],dtype=value.dtype).expand(value.shape)
            elif name=='token_type_ids':restored=torch.zeros(value.shape,dtype=value.dtype)
            else:raise ValueError('Unrestored buffer '+name)
            module.register_buffer(name,restored,persistent=name not in module._non_persistent_buffers_set)
    model.to(device).eval();extra=dict(data['extra']);profile=folder/'inference_settings.json'
    if profile.exists():
        settings=json.loads(profile.read_text());assert settings['weights_updated'] is False
        assert settings['base_checkpoint_sha256']==sha(folder/'model.pt');extra['thresholds']=settings['thresholds']
    assert not any(n.startswith(('pair.','type_emb.','null_parent','active_relations')) for n in model.state_dict())
    return model,extra

def train(root,run='phobert_span_all369_v1',epochs=12,seed=20261117,device='cuda',experiment='configs/experiment_all.json'):
    root=Path(root);torch.set_num_threads(2);random.seed(seed);torch.manual_seed(seed)
    if not torch.cuda.is_available() or not device.startswith('cuda'):raise RuntimeError('Training requires an allocated CUDA GPU')
    folder=root/'checkpoints'/run;output=root/'reports'/run
    if (folder/'model.pt').exists():raise FileExistsError('Preserve existing experiment weights')
    origin=json.loads((root/experiment).read_text());dataset_dir=origin.get('dataset_dir','data/dataset')
    train=read_documents(root/dataset_dir/'train.json');val=read_documents(root/dataset_dir/'val.json');ontology=json.loads((root/'configs/ontology.json').read_text())
    assert 'relations' not in ontology and len(ontology['entities'])==40
    assert all(not d['relations'] and d['input_view']=='docx' for d in train+val)
    assert not {d['group_id'] for d in train}&{d['group_id'] for d in val}
    from .decoding import fit_policy
    counts=collections.Counter(e['label'] for d in train for e in d['entities']);dc=collections.Counter(l for d in train for l in {e['label'] for e in d['entities']});labels=[e['label'] for e in ontology['entities']]
    weights=[min(6.,2*(len(train)/dc[l])**.5) if i>=33 and dc[l] else (2. if i<12 else 1.) for i,l in enumerate(labels)]
    source=root/'resources/phobert-base/pytorch_model.bin'
    assert sha(source)==origin['official_pretrained_sha256'],'Only official unfitted pretrained weights permitted'
    config={'encoder':'phobert','hidden':96,'pretrained_path':str(source.parent),'checkpointing':True,'entity_only':True,'task':'entity_extraction','active_entities':[counts[l]>0 for l in labels],'span_layout_features':True,'quality_v2':True,'quality_v4':True,'balanced_entity_loss':True,'hard_negatives':True,'marker_v6':True,'context_v7':True,'decoder_policy':entity_policy(fit_policy(train,ontology,structured=True,strict_boundaries=True)),'document_recall_weights':weights,'max_span_chars':max(4096,max(e['end']-e['start'] for d in train for e in d['entities'])*2),'decoder_max_entities':max(len(d['entities']) for d in train)+64,'dataset_dir':'data/dataset','accepted_user_dataset':True,'dataset_sha256':dataset_sha256(train,val),'dataset_fingerprint_scheme':'canonical-json-train-val-v1','pretraining':'Official unfitted vinai/phobert-base pinned revision; no E1/E2 or Silver warm start','warm_started_from_legacy_checkpoint':False,'official_pretrained_sha256':origin['official_pretrained_sha256']}
    config.update(dataset_dir=dataset_dir)
    views=train
    if origin.get('train_window_chars'):
        from .document_windows import training_views
        views=training_views(train,origin['train_window_chars'],origin['train_overlap_chars'])
    write_json(output/'TRAIN_VIEWS.json',{'source_documents':len(train),'training_views':len(views),'all_positive_spans_covered':True,'partial_spans_masked_not_negative':True,'max_characters':max(len(d['full_text']) for d in views),'views':[{'document_id':d['document_id'],'start':d.get('source_start',0),'end':d.get('source_end',len(d['full_text'])),'entities':len(d['entities'])} for d in views]})
    topk=max(256,max(collections.Counter(e['label'] for e in d['entities']).most_common(1)[0][1] for d in train)+32)
    model=EntityExtractor(config,ontology).to(device)
    base=list(model.encoder.encoder.parameters());marker=list(model.marker.parameters());used={id(p) for p in base+marker};heads=[p for p in model.parameters() if id(p) not in used]
    optimizer=torch.optim.AdamW([{'params':base,'lr':2e-5},{'params':heads,'lr':3e-4},{'params':marker,'lr':3e-3}],weight_decay=.01)
    total=epochs*len(views);warm=max(1,int(total*.08))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warm,max(.1,.5*(1+math.cos(math.pi*max(0,step-warm)/max(1,total-warm))))))
    scaler=torch.amp.GradScaler('cuda',init_scale=2048);rng=random.Random(seed);best=-1.;history=[];start=time.monotonic()
    hardware={'device':device,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda}
    source_hashes={str(p.relative_to(root)):sha(p) for p in (root/'src').glob('*.py')}
    patience=origin.get('early_stopping_patience',5);minimum_epochs=origin.get('minimum_epochs',8)
    write_json(output/'EXPERIMENT.json',{'run_name':run,'seed':seed,'max_epochs':epochs,'early_stopping':{'patience':patience,'minimum_epochs':minimum_epochs},'train_documents':len(train),'training_views':len(views),'validation_documents':len(val),'source_hashes':source_hashes,'hardware':hardware,'optimizer_lrs':[2e-5,3e-4,3e-3],'config':config})
    thresholds={'boundary_threshold':.5,'span_threshold':.5}
    for epoch in range(1,epochs+1):
        model.train();model._inference_cache=None;order=list(range(len(views)));rng.shuffle(order);losses=collections.Counter();updates=skipped=0;began=time.monotonic()
        for step,index in enumerate(order,1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss,details=model.loss(views[index],rng)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward()
            from .optimization import optimizer_step
            changed=optimizer_step(model,optimizer,scaler,scheduler);updates+=changed;skipped+=not changed
            for k in ['boundary_loss','span_loss','type_competition_loss','attribute_ranking_loss']:losses[k]+=details[k]
            write_json(output/'progress.json',{'status':'TRAINING','epoch':epoch,'max_epochs':epochs,'step':step,'steps':len(views),'loss':float(loss.detach()),'document_id':views[index]['document_id'],'hardware':hardware})
            if step%10==0:print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss.detach())}),flush=True)
        model.eval();metric,_=evaluate_model(model,val,thresholds,topk);score=objective(metric)
        row={'epoch':epoch,'validation':metric['entity_exact_span']['micro'],'macro_f1':metric['entity_exact_span']['macro_f1_supported_labels'],'selection_score':score,'mean_losses':{k:v/len(views) for k,v in losses.items()},'optimizer_updates':updates,'amp_skipped_updates':skipped,'amp_scale':scaler.get_scale(),'seconds':time.monotonic()-began}
        history.append(row);write_json(output/'history.json',history);print('EPOCH_DONE',json.dumps(row),flush=True)
        if score>best+1e-12:
            best=score;save_checkpoint(folder,model,{'epoch':epoch,'seed':seed,'thresholds':thresholds,'decoder_topk':topk,'selected_on':'Validation mean entity Micro/Macro F1','official_pretrained_only':True})
        if epoch>=minimum_epochs and epoch-history[max(range(len(history)),key=lambda i:history[i]['selection_score'])]['epoch']>=patience:break
    del model,optimizer,scaler,base,heads,marker;torch.cuda.empty_cache()
    model,extra=load_checkpoint(folder,device);model._inference_cache={};best_t=dict(thresholds);best_score=-1.;trials=[]
    for b in [.35,.5,.65]:
        for s in [.35,.5,.75]:
            t={'boundary_threshold':b,'span_threshold':s};metric,_=evaluate_model(model,val,t,topk);score=objective(metric);trials.append({'thresholds':t,'selection_score':score,'micro':metric['entity_exact_span']['micro']})
            if score>best_score:best_score=score;best_t=t
    report={'status':'TRAINED_ENTITY_ONLY_ON_GPU','run_name':run,'train_documents':len(train),'validation_documents':len(val),'test_documents':0,'epochs_completed':len(history),'selected_epoch':extra['epoch'],'seed':seed,'hardware':hardware,'source_hashes':source_hashes,'dataset_sha256':config['dataset_sha256'],'dataset_fingerprint_scheme':config['dataset_fingerprint_scheme'],'official_pretrained_only':True,'fitted_parent_checkpoint_used':False,'relation_parameters':0,'independent_test_claimed':False,'threshold_trials':trials,'thresholds':best_t,'results':{},'seconds':time.monotonic()-start}
    write_json(folder/'inference_settings.json',{'weights_updated':False,'base_checkpoint_sha256':sha(folder/'model.pt'),'selection_split':'Validation hyperparameters only','thresholds':best_t})
    model._inference_cache=None
    for split,docs in [('train',train),('val',val)]:
        metric,pred=evaluate_model(model,docs,best_t,topk);report['results'][split]=metric;write_json(output/'predictions'/(split+'.json'),pred)
    write_json(output/'RESULTS.json',report);print('TRAIN_COMPLETE',json.dumps({s:m['entity_exact_span']['micro'] for s,m in report['results'].items()}),flush=True)
    return report
