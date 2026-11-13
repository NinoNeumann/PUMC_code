import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root='E:/CodexPorjects/PUMC-rebuttal';
const source=path.join(root,'results/rcs_ridge_10x5/20261102_v1/fold_first_cindex_20261103');
const outputDir=path.join(root,'outputs/cacox_es_available_20261103');
const outputPath=path.join(outputDir,'cacox_es_vs_rcs_50fold_available_metrics_auc_pvalues.xlsx');
const data=JSON.parse(await fs.readFile(path.join(source,'available_metrics_workbook_data.json'),'utf8'));
await fs.mkdir(path.join(outputDir,'previews'),{recursive:true});
const wb=Workbook.create();
const navy='#17324F', pale='#EEF4F8';
function col(n){let x=n+1,s='';while(x){x--;s=String.fromCharCode(65+x%26)+s;x=Math.floor(x/26)}return s}
function label(x){return String(x).replaceAll('E_selected','cacox（es）').replaceAll('rcs_ridge_cox','RCS＋ridge-Cox').replaceAll('E_minus_RCS','cacox（es）−RCS').replaceAll('repeat_id','重复轮次').replaceAll('fold','折次').replaceAll('imputation','插补').replaceAll('horizon_days','时间点（天）').replaceAll('CI95_lower','95%CI下限').replaceAll('CI95_upper','95%CI上限').replaceAll('Brier_IPCW','IPCW Brier').replaceAll('difference','差值').replaceAll('net_benefit','净获益').replaceAll('observed_events','事件数').replaceAll('predicted_survival','平均预测生存率').replaceAll('observed_KM_survival','KM观察生存率').replaceAll('patients','人数').replaceAll('threshold','风险阈值').replaceAll('quintile','五分位组').replaceAll('model','方法').replaceAll('FPR','假阳性率').replaceAll('TPR','真阳性率');}
function table(name,obj,{color=navy}={}){
  const sh=wb.worksheets.add(name); sh.showGridLines=false;sh.tabColor=color;
  const {columns,rows}=obj,n=columns.length,last=col(n-1);
  sh.getRange(`A1:${last}1`).values=[columns.map(label)];
  sh.getRange(`A1:${last}1`).format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:43,wrapText:true,verticalAlignment:'center'};
  for(let start=0;start<rows.length;start+=2000){let part=rows.slice(start,start+2000);sh.getRange(`A${start+2}:${last}${start+part.length+1}`).values=part;}
  for(let j=0;j<n;j++){
    const c=col(j),v=columns[j];
    sh.getRange(`${c}:${c}`).format.columnWidth=Math.min(28,Math.max(13,label(v).length*1.1));
    if(rows.length&&j>0&&!['model','stratum'].includes(v)){
      const integer=['repeat_id','fold','imputation','imputations','horizon_days','patients','events','observed_events','cases','controls','quintile'].includes(v);
      sh.getRange(`${c}2:${c}${rows.length+1}`).setNumberFormat(integer?'0':'0.00000');
    }
  }
  sh.freezePanes.freezeRows(1);
  return sh;
}
const sh=wb.worksheets.add('总体对比');sh.showGridLines=false;sh.tabColor='#1B7652';
sh.getRange('A1:F1').merge();sh.getRange('A1').values=[['cacox（es） 与 RCS＋ridge-Cox｜10 次 × 5 折 × 20 份插补']];
sh.getRange('A1:F1').format={fill:navy,font:{name:'Aptos',size:16,bold:true,color:'#FFFFFF'},rowHeight:39};
sh.getRange('A3:F3').values=[['指标','方法','时间点（天）','估计值','95%CI下限','95%CI上限']];
sh.getRange('A3:F3').format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:28};
sh.getRange(`A4:F${data.summary.length+3}`).values=data.summary;
sh.getRange(`D4:F${data.summary.length+3}`).setNumberFormat('0.00000');
const r=5+data.summary.length;
sh.getRange(`A${r}:F${r}`).values=[['配对差值','时间点（天）','估计差值','95%CI下限','95%CI上限','Rubin p值']];
sh.getRange(`A${r}:F${r}`).format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:28};
sh.getRange(`A${r+1}:F${r+data.comparison.length}`).values=data.comparison;
sh.getRange(`C${r+1}:F${r+data.comparison.length}`).setNumberFormat('0.000000');
const note=r+data.comparison.length+2;
sh.getRange(`A${note}:B${note+5}`).values=[
  ['50 折','每行是一轮中的一折；各折先汇总 20 份插补。'],
  ['方法名称','cacox（es）是原实验 E_selected 的展示名称；历史计算文件未改名。'],
  ['曲线范围','ROC 与 DCA 含 50 折逐折坐标；图页提供总体展示。'],
  ['生存校准','仅现成总体校准曲线；没有现成的 50 条逐折校准曲线。'],
  ['IBS','入组至第 3 年 IBS 尚未完成核验；本工作簿不含 IBS。'],
  ['推断限制','同一 1140 名患者在重复折次和插补中相关；不能当成独立验证样本。'],
];
sh.getRange(`A${note}:A${note+5}`).format={fill:pale,font:{name:'Aptos',size:10,bold:true,color:navy}};
sh.getRange('A:A').format.columnWidth=24;sh.getRange('B:B').format.columnWidth=57;sh.getRange('C:F').format.columnWidth=18;sh.freezePanes.freezeRows(3);

const sheets=[
  ['50折指标','fold'],['AUC逐折详情','auc_detail'],['1000模型明细','unit'],
  ['20插补总体','imputation'],['ROC逐折坐标','roc_fold'],['DCA逐折坐标','dca_fold'],
  ['ROC总体坐标','roc_overall'],['DCA总体坐标','dca_overall'],['生存校准总体','calibration_overall'],
];
for(const [name,key] of sheets)table(name,data[key],{color:key.includes('roc')?'#2A65A0':key.includes('dca')?'#8261A1':'#28796D'});

const figures=wb.worksheets.add('曲线图');figures.showGridLines=false;
figures.getRange('A1').values=[['三种曲线：总体展示（每折坐标见相应数据表）']];
figures.getRange('A1').format={fill:navy,font:{name:'Aptos',size:14,bold:true,color:'#FFFFFF'},rowHeight:32};
figures.getRange('A2').values=[['生存校准图仅有总体结果；不能解读为 50 条逐折曲线。']];
const pngs=[['ROC 曲线','roc_3_horizons.png',2],['DCA 曲线','dca_3_horizons.png',28],['生存校准曲线','survival_calibration.png',54]];
for(const [title,file,row] of pngs){
  figures.getRange(`A${row}`).values=[[title]];
  const buf=await fs.readFile(path.join(source,'figures',file));
  figures.images.add({dataUrl:`data:image/png;base64,${buf.toString('base64')}`,anchor:{from:{row:row,col:0},extent:{widthPx:900,heightPx:340}}});
}
figures.getRange('A:A').format.columnWidth=35;

const notes=wb.worksheets.add('口径与来源');notes.showGridLines=false;
notes.getRange('A1:B1').values=[['项目','说明']];notes.getRange('A1:B1').format={fill:navy,font:{name:'Aptos',bold:true,color:'#FFFFFF'}};
const details=[
  ['研究设计','同一 1140 人；10 次重复 5 折；每折 20 份 R MICE 插补；两方法同折配对。'],
  ['C-index','50折指标中的C-index及区间来自fold_pooled_cindex.csv。'],
  ['AUC','第365、730、1095天的时间依赖AUC；逐折估计、区间和p值来自auc_fold_pooled.csv。'],
  ['Brier','各折各时间点IPCW Brier由同折20份插补结果取均值；越低越好。'],
  ['ROC','ROC逐折坐标含50折×2方法×3时间点×201个FPR点；曲线坐标不是另行训练所得。'],
  ['DCA','DCA逐折坐标含50折×2方法×3时间点×30个风险阈值，阈值为0.01至0.30。'],
  ['校准','仅保存总体五分位生存校准，2方法×3时间点×5组，不是逐折校准。'],
  ['IBS','0–1095天IBS未完成；以前三时点的1–3年近似IBS不纳入本表。'],
  ['名称映射','cacox（es）=历史输出中的E_selected；RCS＋ridge-Cox=历史输出中的rcs_ridge_cox。'],
  ['来源目录',data.source],
];
notes.getRange(`A2:B${details.length+1}`).values=details;
notes.getRange('A:A').format.columnWidth=22;notes.getRange('B:B').format.columnWidth=105;notes.getRange(`B2:B${details.length+1}`).format.wrapText=true;
notes.freezePanes.freezeRows(1);

wb.recalculate();
for(const [name,key] of sheets){
  const end=col(Math.min(data[key].columns.length,7)-1);
  try{const p=await wb.render({sheetName:name,range:`A1:${end}7`,scale:1,format:'png'});await fs.writeFile(path.join(outputDir,'previews',`${key}.png`),new Uint8Array(await p.arrayBuffer()));}
  catch(e){console.log('preview warning',name,String(e));}
}
for(const [name,range] of [['总体对比','A1:F20'],['口径与来源','A1:B11']]){
  const p=await wb.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(outputDir,'previews',`${name}.png`),new Uint8Array(await p.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(outputPath);
console.log(JSON.stringify({outputPath,counts:Object.fromEntries(sheets.map(([,k])=>[k,data[k].rows.length]))}));
