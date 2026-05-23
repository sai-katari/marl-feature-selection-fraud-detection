import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load and split the same way as the main training script
df = pd.read_csv('data.csv', index_col=0, encoding='UTF-8')
X = df.drop(['class'], axis=1)
y = pd.DataFrame([0 if x == 0 else 1 for x in df['class']], columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
oversample = RandomOverSampler(sampling_strategy='minority', random_state=123)
X_res, y_res = oversample.fit_resample(X_train, y_train)

print("Data loaded. Training models for plots...")

def get_xgb():
    return xgb.XGBClassifier(
        random_state=42, eval_metric='logloss',
        n_estimators=100, max_depth=6, n_jobs=-1
    )

# Feature subset the MARL agents converged on after 500 episodes
marl_features = [
    'V2', 'V3', 'V4', 'V5', 'V8', 'V10', 'V11', 'V12', 'V13',
    'V14', 'V18', 'V19', 'V20', 'V21', 'V22', 'V24', 'V25', 'V28', 'Amount'
]

# Train baseline on all 29 features
print("Training baseline (all features)...")
clf_all = get_xgb()
clf_all.fit(X_res, y_res.values.ravel())
pred_all = clf_all.predict(X_test)
cm_all = confusion_matrix(y_test.values.ravel(), pred_all)
f1_all = f1_score(y_test.values.ravel(), pred_all, average='macro')

# Train on MARL-selected 19 features
print("Training MARL model (19 features)...")
clf_marl = get_xgb()
clf_marl.fit(X_res[marl_features], y_res.values.ravel())
pred_marl = clf_marl.predict(X_test[marl_features])
cm_marl = confusion_matrix(y_test.values.ravel(), pred_marl)
f1_marl = f1_score(y_test.values.ravel(), pred_marl, average='macro')

print(f"Baseline F1: {round(f1_all, 4)}")
print(f"MARL F1: {round(f1_marl, 4)}")
print("Building plots...")

# F1 scores recorded per episode from the actual 500-episode training run
episode_f1 = {
    0:47.62,1:43.94,2:47.62,3:52.99,4:52.99,5:52.99,6:52.99,7:52.99,
    8:52.99,9:61.00,10:75.38,11:84.04,12:85.89,13:85.89,14:85.89,
    15:84.98,16:85.89,17:85.89,18:92.69,19:92.69,20:92.69,21:92.77,
    22:93.00,23:93.00,24:92.23,25:93.00,26:93.00,27:93.77,28:93.77,
    29:94.32,30:94.32,31:93.71,32:93.84,33:93.84,34:94.38,35:94.38,
    36:94.38,37:94.14,38:94.38,39:94.38,40:94.38,41:94.38,42:94.38,
    43:94.38,44:94.01,45:94.38,46:94.38,47:94.38,48:94.38,49:94.38,
    50:94.19,51:94.38,52:94.38,53:94.38,54:94.38,55:94.14,56:94.38,
    57:94.38,58:94.38,59:94.38,60:94.38,61:94.01,62:94.38,63:94.38,
    64:94.50,65:94.50,66:94.50,67:94.50,68:94.50,69:94.08,70:91.83,
    71:94.50,72:93.47,73:93.23,74:94.50,75:94.50,76:94.50,77:94.32,
    78:94.32,79:93.91,80:94.50,81:94.12,82:94.50,83:94.50,84:94.50,
    85:94.01,86:94.50,87:94.50,88:94.44,89:94.01,90:94.50,91:94.50,
    92:94.50,93:94.50,94:94.50,95:94.50,96:94.50,97:94.61,98:94.86,
    99:94.25,100:93.77,101:94.86,102:94.86,103:94.86,104:94.86,
    105:94.86,106:94.86,107:94.86,108:94.86,109:94.86,110:94.86,
    111:93.47,112:94.56,113:94.61,114:94.86,115:94.86,116:94.86,
    117:94.38,118:94.80,119:94.80,120:94.80,121:94.80,122:94.80,
    123:94.80,124:94.86,125:94.86,126:95.15,127:94.86,128:94.91,
    129:94.86,130:95.15,131:94.86,132:94.32,133:94.86,134:95.10,
    135:94.86,136:94.97,137:93.47,138:94.86,139:94.86,140:95.15,
    141:94.86,142:94.86,143:94.86,144:94.86,145:95.15,146:95.10,
    147:95.65,148:94.86,149:94.86,150:94.86,151:94.80,152:94.86,
    153:94.08,154:94.86,155:94.86,156:94.86,157:94.86,158:94.86,
    159:94.86,160:94.86,161:94.86,162:94.86,163:94.86,164:94.86,
    165:93.77,166:94.86,167:94.86,168:94.86,169:95.40,170:95.40,
    171:94.86,172:95.40,173:95.40,174:95.40,175:94.61,176:94.86,
    177:95.40,178:95.40,179:94.32,180:95.40,181:94.56,182:95.40,
    183:95.40,184:95.40,185:95.40,186:95.40,187:95.45,188:95.40,
    189:95.40,190:95.40,191:95.40,192:95.40,193:95.40,194:95.40,
    195:95.40,196:95.40,197:95.40,198:95.40,199:95.40,200:94.91,
    201:94.91,202:95.40,203:95.40,204:94.32,205:95.40,206:95.40,
    207:95.40,208:95.40,209:95.40,210:95.40,211:95.40,212:95.40,
    213:95.40,214:95.40,215:94.80,216:94.56,217:95.40,218:95.40,
    219:95.65,220:94.86,221:94.86,222:94.86,223:94.86,224:94.86,
    225:94.86,226:94.86,227:94.86,228:94.86,229:94.86,230:94.86,
    231:95.05,232:95.69,233:95.69,234:95.15,235:95.69,236:95.69,
    237:95.69,238:95.69,239:95.69,240:95.69,241:95.69,242:93.77,
    243:95.45,244:95.69,245:94.19,246:95.69,247:93.77,248:95.69,
    249:95.40,250:95.69,251:95.69,252:95.69,253:95.69,254:94.91,
    255:95.69,256:95.69,257:94.91,258:95.69,259:95.69,260:94.08,
    261:95.69,262:95.69,263:95.69,264:95.69,265:95.69,266:94.32,
    267:94.19,268:95.45,269:94.20,270:95.69,271:95.69,272:95.69,
    273:95.40,274:95.69,275:95.69,276:95.69,277:95.69,278:95.69,
    279:95.69,280:95.69,281:95.69,282:95.69,283:95.69,284:95.69,
    285:95.69,286:95.69,287:95.69,288:95.69,289:95.69,290:95.69,
    291:95.69,292:95.05,293:94.86,294:95.69,295:95.69,296:95.69,
    297:95.69,298:95.69,299:95.69,300:95.69,301:95.69,302:95.69,
    303:95.69,304:95.69,305:95.69,306:95.69,307:95.69,308:95.69,
    309:95.69,310:95.69,311:94.56,312:95.69,313:95.69,314:95.20,
    315:95.69,316:95.69,317:95.69,318:95.69,319:95.69,320:95.69,
    321:95.69,322:95.69,323:95.69,324:95.69,325:95.69,326:95.69,
    327:95.69,328:95.69,329:95.69,330:95.69,331:95.69,332:95.69,
    333:95.69,334:95.69,335:95.69,336:95.69,337:95.69,338:95.69,
    339:95.69,340:95.69,341:95.69,342:95.69,343:95.69,344:95.69,
    345:95.69,346:95.69,347:95.69,348:95.69,349:95.69,350:95.69,
    351:95.69,352:95.69,353:95.69,354:95.69,355:95.69,356:95.69,
    357:95.69,358:95.69,359:95.69,360:93.91,361:95.69,362:95.69,
    363:95.69,364:95.69,365:95.69,366:95.69,367:95.69,368:95.69,
    369:95.69,370:95.98,371:95.69,372:95.69,373:95.69,374:95.69,
    375:95.69,376:95.69,377:95.69,378:95.69,379:95.69,380:95.69,
    381:95.69,382:95.69,383:95.69,384:95.69,385:95.69,386:95.69,
    387:95.69,388:95.69,389:95.69,390:95.69,391:95.69,392:95.69,
    393:95.69,394:95.69,395:95.69,396:95.69,397:95.69,398:95.69,
    399:95.69,400:95.69,401:95.69,402:94.56,403:95.69,404:95.69,
    405:95.69,406:95.69,407:95.69,408:95.69,409:95.69,410:95.69,
    411:95.69,412:95.69,413:95.69,414:95.69,415:95.69,416:95.69,
    417:95.69,418:95.69,419:95.69,420:95.69,421:95.69,422:95.69,
    423:95.69,424:95.69,425:95.69,426:95.69,427:95.69,428:95.69,
    429:95.69,430:95.69,431:95.69,432:95.69,433:95.69,434:95.69,
    435:95.69,436:95.69,437:95.69,438:95.69,439:95.69,440:95.69,
    441:95.69,442:95.40,443:95.69,444:95.69,445:95.69,446:95.69,
    447:93.77,448:95.69,449:95.69,450:95.69,451:95.69,452:95.69,
    453:95.69,454:95.69,455:95.69,456:95.69,457:95.69,458:95.69,
    459:93.97,460:95.69,461:95.69,462:95.69,463:95.69,464:95.69,
    465:95.69,466:95.69,467:95.69,468:95.69,469:95.69,470:95.69,
    471:95.69,472:95.69,473:95.69,474:95.69,475:95.69,476:95.69,
    477:95.69,478:94.19,479:95.69,480:95.69,481:95.69,482:95.69,
    483:95.69,484:95.05,485:95.69,486:95.69,487:95.69,488:95.69,
    489:95.69,490:95.69,491:95.69,492:95.69,493:95.69,494:95.69,
    495:95.69,496:95.69,497:95.69,498:95.69,499:95.69
}

episodes = list(range(500))
f1_curve  = [episode_f1[e] for e in episodes]

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
c_marl = '#1565C0'

# Plot 1: Training curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(episodes, f1_curve, color=c_marl, linewidth=1.5, alpha=0.85, label='MARL F1')
ax.axhline(y=f1_all * 100, color='red', linestyle='--',
           linewidth=1.3, label=f'All Features Baseline ({round(f1_all, 4)})')
ax.axhline(y=f1_marl * 100, color='green', linestyle='--',
           linewidth=1.3, label=f'MARL Converged ({round(f1_marl, 4)})')
ax.set_xlabel('Episode')
ax.set_ylabel('Macro F1 Score (%)')
ax.set_title('MARL Training Curve — Credit Card Fraud Detection', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)
ax.set_ylim(40, 100)
plt.tight_layout()
plt.savefig('plot_training_curve.png', dpi=150, bbox_inches='tight')
print("Saved: plot_training_curve.png")
plt.close()

# Plot 2: Side-by-side confusion matrices
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Confusion Matrices — All Features vs MARL Selected',
              fontsize=13, fontweight='bold')
for ax, cm, title in zip(
    axes2,
    [cm_all, cm_marl],
    ['All Features — 29 (Baseline)', 'MARL Selected — 19 Features']
):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'],
                linewidths=0.5, cbar=False, annot_kws={'size': 13})
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: plot_confusion_matrix.png")
plt.close()

# Plot 3: Feature importance — V14 dominates by a wide margin
print("Computing feature importance...")
importances = clf_marl.feature_importances_
feat_imp = pd.Series(importances, index=marl_features).sort_values(ascending=True)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.barh(feat_imp.index, feat_imp.values,
         color=c_marl, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Feature Importance (XGBoost)')
ax3.set_title('MARL Selected Features — Importance Ranking', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: plot_feature_importance.png")
plt.close()

print("\nAll 3 plots saved.")
