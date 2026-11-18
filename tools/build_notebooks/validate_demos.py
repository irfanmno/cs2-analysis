import pandas as pd
pd.set_option('display.width',250)
r=pd.read_parquet('data/processed/rounds.parquet'); pr=pd.read_parquet('data/processed/player_rounds.parquet')
m=pd.read_parquet('data/processed/matches.parquet').set_index('match_id')
rows=[]
for mid,g in r.groupby('match_id'):
    me=pr[(pr.match_id==mid)&pr.is_me]; n=g['round'].nunique()
    rows.append(dict(match_id=mid[:10],rounds=n,f_rounds=int(m.loc[mid,'rounds']),wins=int(g.my_team_won.sum()),f_wins=int(m.loc[mid,'my_score']),
        K=int(me.kills.sum()),fK=int(m.loc[mid,'kills']),D=int(me.deaths.sum()),fD=int(m.loc[mid,'deaths']),A=int(me.assists.sum()),fA=int(m.loc[mid,'assists']),
        ADR=round(me.damage.sum()/n,1),fADR=m.loc[mid,'adr'],rmin=int(g['round'].min()),rmax=int(g['round'].max())))
c=pd.DataFrame(rows)
bad=c[(c.rounds!=c.f_rounds)|(c.wins!=c.f_wins)|(c.K!=c.fK)|(c.D!=c.fD)|(c.A!=c.fA)|((c.ADR-c.fADR).abs()>0.5)]
print(f'{len(c)} matches, {len(c)-len(bad)} match FACEIT exactly; rounds {c.rounds.sum()} vs {c.f_rounds.sum()}'); print(bad.to_string(index=False))
