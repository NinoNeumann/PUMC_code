import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = 'E:/CodexPorjects/PUMC-rebuttal';
const dataPath = path.join(root, 'results/rcs_ridge_10x5/20261102_v1/fold_first_cindex_20261103/es_workbook_data.json');
const outputDir = path.join(root, 'outputs/cacox_es_10x5x20_20261103');
const outputPath = path.join(outputDir, 'cacox_es_10x5x20_results.xlsx');
const previewDir = path.join(outputDir, 'previews');
const data = JSON.parse(await fs.readFile(dataPath, 'utf8'));
await fs.mkdir(previewDir, {recursive:true});
const wb = Workbook.create();

const friendly = s => String(s)
  .replaceAll('E_selected', 'cacox（es）')
  .replaceAll('rcs_ridge_cox', 'RCS＋ridge-Cox')
  .replaceAll('E_minus_RCS', 'cacox（es）−RCS')
  .replaceAll('repeat_id', '重复轮次')
  .replaceAll('fold', '折次')
  .replaceAll('imputation', '插补序号')
  .replaceAll('horizon_days', '时间点（天）')
  .replaceAll('difference', '差值')
  .replaceAll('CI95_lower', '95%CI下限')
  .replaceAll('CI95_upper', '95%CI上限')
  .replaceAll('Brier_IPCW', 'IPCW Brier')
  .replaceAll('IBS_approx', '三点近似IBS')
  .replaceAll('net_benefit', '净获益');

const colLetter = n => { let x=n+1, s=''; while (x) { x--; s=String.fromCharCode(65+x%26)+s; x=Math.floor(x/26); } return s; };
const navy='#17324F', blue='#DDEBF6', light='#F2F6FA', grey='#526273';
function makeSheet(name, columns, rows, {tab=navy, numberDigits=5}={}) {
  const sh=wb.worksheets.add(name);
  sh.showGridLines=false; sh.tabColor=tab;
  const last=colLetter(columns.length-1);
  const headers=columns.map(friendly);
  sh.getRange(`A1:${last}1`).values=[headers];
  sh.getRange(`A1:${last}1`).format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:43,wrapText:true,verticalAlignment:'center'};
  if(rows.length) sh.getRange(`A2:${last}${rows.length+1}`).values=rows;
  sh.getRange(`A1:${last}${Math.max(rows.length+1,2)}`).format.font={name:'Aptos',size:10};
  sh.getRange(`A1:${last}1`).format.font={name:'Aptos',size:10,bold:true,color:'#FFFFFF'};
  sh.getRange(`A1:${last}1`).format.fill=navy;
  for(let c=0;c<columns.length;c++) {
    const col=colLetter(c), key=columns[c];
    sh.getRange(`${col}:${col}`).format.columnWidth=Math.min(28,Math.max(14,headers[c].length*1.05));
    if(rows.length && !['model','stratum'].includes(key)) {
      const integer = ['repeat_id','fold','imputation','imputations','horizon_days','quintile','patients','events','observed_events','cases','controls'].includes(key);
      sh.getRange(`${col}2:${col}${rows.length+1}`).setNumberFormat(integer?'0':`0.${'0'.repeat(numberDigits)}`);
    }
  }
  if(rows.length) {
    sh.getRange(`A2:${Math.min(columns.length,3)>1?colLetter(Math.min(columns.length,3)-1):'A'}${rows.length+1}`).format.fill=light;
  }
  sh.freezePanes.freezeRows(1);
  return sh;
}

const summary=wb.worksheets.add('总览');
summary.showGridLines=false;summary.tabColor='#1D6F42';
summary.getRange('A1:F1').merge();
summary.getRange('A1').values=[['cacox（es）与 RCS＋ridge-Cox：10次×5折×20份插补']];
summary.getRange('A1:F1').format={fill:navy,font:{name:'Aptos',size:16,bold:true,color:'#FFFFFF'},rowHeight:38};
summary.getRange('A3:F3').values=[['指标','方法','时间点（天）','估计值','95%CI下限','95%CI上限']];
summary.getRange('A3:F3').format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:27};
summary.getRange(`A4:F${3+data.summary.length}`).values=data.summary;
summary.getRange(`D4:F${3+data.summary.length}`).setNumberFormat('0.00000');
const cr=5+data.summary.length;
summary.getRange(`A${cr}:F${cr}`).values=[['配对差值','时间点（天）','估计差值','95%CI下限','95%CI上限','Rubin p值']];
summary.getRange(`A${cr}:F${cr}`).format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:27};
summary.getRange(`A${cr+1}:F${cr+data.comparison.length}`).values=data.comparison;
summary.getRange(`C${cr+1}:F${cr+data.comparison.length}`).setNumberFormat('0.000000');
const nr=cr+data.comparison.length+2;
summary.getRange(`A${nr}:B${nr+6}`).values=[
  ['范围','1140名患者；10次重复×5外层折；每折20份R MICE插补'],
  ['模型数','每个方法1000个同折同插补外层模型；50个折内合并结果'],
  ['显示名称','cacox（es）；原始结果文件中的方法ID为 E_selected'],
  ['50折汇总','同一折的20份插补先合并；一折一行'],
  ['20插补总体','同一插补编号先跨50折汇总，再按Rubin规则计算总体估计'],
  ['近似IBS','仅365、730、1095天三个时间点的数值积分，并非连续时间IBS'],
  ['推断范围','同队列反复开发与评估；1000个插补模型相关，不能视为1000个独立样本'],
];
summary.getRange(`A${nr}:A${nr+6}`).format={fill:blue,font:{name:'Aptos',size:10,bold:true,color:navy}};
summary.getRange('A:A').format.columnWidth=24;
summary.getRange('B:B').format.columnWidth=54;
summary.getRange('C:F').format.columnWidth=19;
summary.freezePanes.freezeRows(3);

const configs=[
  ['50折汇总','fold',navy],
  ['1000模型明细','unit','#2B579A'],
  ['20插补总体','imputation','#28796D'],
  ['DCA净获益','dca','#694B8A'],
  ['生存校准','calibration','#A15D2C'],
  ['风险分层','strata','#A15D2C'],
  ['ROC坐标','roc','#2B579A'],
];
for(const [name,key,tab] of configs) makeSheet(name,data[key].columns,data[key].rows,{tab});

const notes=wb.worksheets.add('口径与来源');
notes.showGridLines=false;
notes.getRange('A1:B1').values=[['项目','说明']];
notes.getRange('A1:B1').format={fill:navy,font:{name:'Aptos',size:10,bold:true,color:'#FFFFFF'},rowHeight:27};
const noteRows=[
  ['名称对应','本工作簿将原始结果ID E_selected 展示为 cacox（es）；仅改变标签，不修改历史实验或任何计算结果。'],
  ['比较方法','RCS＋ridge-Cox 使用同一批外层折、插补、变量筛选结果和测试患者。'],
  ['C-index','1000模型明细每行对应一轮、一折、一份插补；50折汇总将同折20份插补合并。'],
  ['AUC','时间依赖AUC分别在365、730、1095天计算；ROC坐标为相应时间点的汇总曲线。'],
  ['IPCW Brier','逆概率删失加权Brier分数，越低越好。'],
  ['三点近似IBS','365至1095天仅用三时点进行数值积分，不能等同完整密集时间网格IBS。'],
  ['DCA','DCA净获益为汇总估计；阈值列为风险阈值。'],
  ['生存校准','每个时间点按预测风险五分位分组；展示预测生存率和Kaplan–Meier观察生存率。'],
  ['统计解释','Rubin p值与95%CI按原统计流程；同一患者及20份插补存在相关性。'],
  ['来源目录',data.source],
  ['来源明细','final/model_cindex.csv、final/metrics_by_imputation.csv、fold_first_cindex_20261103目录下的各指标CSV。'],
  ['日期','2026-11-03'],
];
notes.getRange(`A2:B${noteRows.length+1}`).values=noteRows;
notes.getRange('A:A').format.columnWidth=24;
notes.getRange('B:B').format.columnWidth=115;
notes.getRange(`B2:B${noteRows.length+1}`).format.wrapText=true;
notes.freezePanes.freezeRows(1);

wb.recalculate();
const check=await wb.inspect({kind:'region',sheetId:summary.id,range:'A3:F8',maxChars:2000,tableMaxRows:7,tableMaxCols:6});
console.log('summary inspect',JSON.stringify(check).slice(0,2200));
for(const [name,key] of configs) {
  const sh=wb.worksheets.getItem(name);
  const end=colLetter(Math.min(data[key].columns.length,8)-1);
  try {
    const png=await wb.render({sheetName:name,range:`A1:${end}${Math.min(8,data[key].rows.length+1)}`,scale:1,format:'png'});
    await fs.writeFile(path.join(previewDir,`${key}.png`),new Uint8Array(await png.arrayBuffer()));
  } catch(e) { console.log('render warning',name,String(e)); }
}
for(const name of ['总览','口径与来源']) {
  try {
    const png=await wb.render({sheetName:name,range:name==='总览'?'A1:F22':'A1:B12',scale:1,format:'png'});
    await fs.writeFile(path.join(previewDir,`${name}.png`),new Uint8Array(await png.arrayBuffer()));
  } catch(e) { console.log('render warning',name,String(e)); }
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({outputPath,sheets:wb.worksheets.items.map(x=>x.name),counts:Object.fromEntries(configs.map(([,k])=>[k,data[k].rows.length]))}));
