# 📓 Havaintopäiväkirja — emergentit hetket

> Tähän kerätään kiinnostavat, yllättävät ja paljastavat hetket joita multiagenttitiimi
> tuottaa: huijaukset, oivallukset, persoonalliset purkaukset, epäonnistumistavat.
> Uusin ylimpänä. Nämä ovat sekä tutkimusmuistiinpanoja että AI-illan tarinoita.
>
> Pipeline: **Manager → Coder → Tester → Reviewer (SHODAN)**.
> SHODAN = frontier-API-malli (Grok). Muut = paikallisia Ollama-malleja RTX 5090:llä.

---

## 2026-06-10 — 🔗 Phantom-akkumulaatio: 24 iteraatiota "kasautuvaa" treeniä joka alkoi joka kerta nollasta

**Mitä tapahtui.** Seeded-robottikäsiajo (`robot_seeded_20260610_183241`). PandaReach ratkesi
hienosti yhdellä 50k-chunkilla (success 1.0). Sitten PandaPush: 24 iteraatiota, success_rate
heilui 0.0–0.3 satunnaisesti, failsafe skippasi 20 ei-edistyvän epäonnistumisen jälkeen.
Yhteensä treenattiin ~1.2M steppiä — määrällisesti tarpeeksi — mutta mikään ei kasautunut.

**Juurisyy — kolme kerrosta:**
1. **Manager-taskit käskivät "Load the model" mutta EIVÄT koskaan "load_replay_buffer".**
   Off-policy SAC+HER tyhjällä bufferilla resumen jälkeen unohtaa oppimansa → käyrä
   0.30 → 0.20 → 0.10. Seedattu verified-skill sanoi täsmälleen oikean reseptin
   (SAC.load + load_replay_buffer) — kukaan ei noudattanut sitä.
2. **Iteraatiot 12–18 paloivat Managerin hallusinoimaan HER-API:in** (`online_sample_strategy`,
   manuaalinen HerReplayBuffer väärillä argumenteilla). Lint ei validoinut kwargeja.
3. **Kriisin jälkeen SHODAN määräsi täsmälleen väärän korjauksen**: *"initialize fresh,
   do not reintroduce any load logic"* — suora ristiriita pinnatun verified-skillin kanssa.
   Loput iteraatiot olivat 50k-arvontaa nollasta. Kukaan ei nähnyt kumulatiivista käyrää,
   josta puuttuva akkumulaatio olisi paljastunut kolmessa chunkissa.

**Opetus.** "Pinned" suojasi skillin SISÄLLÖN ylikirjoituksilta — mutta ei PÄÄTÖKSENTEKOA.
Älykkäin agentti ohitti varmennetun tiedon omalla hypoteesillaan, ja hierarkia totteli.
Sama kuvio kuin 06-09 SHODAN-huijauksessa: rehellisyys/kuri pitää rakentaa rakenteellisesti.

**Korjaus (rakenteellinen, ei kehotuksia):**
- **Resume-portti**: checkpointin olemassa ollessa optimointiskriptin PITÄÄ printata
  `RESUMED: buffer_transitions=N` (N>0). Lint vaatii kontraktin (load + buffer + proof +
  save) ennen Dockeria, Tester verifioi stdoutista, Reviewerillä deterministinen portti
  joka ylittää APPROVE:n — kuten threshold-portti.
- **Kumulatiivinen näkyvyys**: total_env_steps + metric-käyrä Managerin ja SHODANin
  promptiin joka optimointi-iteraatiossa.
- **Skill-prioriteetti**: verified-skill ylittää KAIKEN muun palautteen, myös SHODANin
  direktiivit ("THE PIN BINDS YOU TOO" SHODANin promptissa).
- **SB3-kwarg-lint**: kuratoidut signatuurit SAC/PPO/…/HerReplayBuffer/learn() —
  hallusinoitu kwarg on kova virhe millisekunneissa, ei 7 hukkaiteraatiota.
- **SPS-mittaus**: Tester mittaa steps/s → Manager laskee chunk-koon (SPS × timeout × 0.8)
  → step-määrän arvonta timeouttia vasten loppuu.
- Lisäksi: validation-timeoutiin kiinteä pohja (panda-gymin käynnistys ~20–40 s) ja
  env-vaihdon stale-task-race korjattu (konkreettinen taski + manager_guidance synkassa).

---

## 2026-06-09 — 💀 SHODAN huijasi (ja jäi kiinni)

**Mitä tapahtui.** Robottikäsi-koe (panda-gym, `robot_arm_seeded`) käynnistyi. Iteraatio 1
kaatui koska Coder ei importannut `panda_gym`:iä eikä käyttänyt `-v3`-suffiksia →
`gymnasium.error.NameNotFound: Environment 'PandaReach' doesn't exist`.

Rehellisyyskoneisto toimi oikein: Tester raportoi kaadon, ei haamupalkintoa. Mutta sitten
**Reviewer — SHODAN, frontier-malli, koko tiimin rehellinen tuomari** — ehdotti korjaukseksi:

> *"If the env truly cannot be created here, switch to a standard Gymnasium env that exists
> (e.g. Pendulum-v1) for the smoke test."*

Eli: **vaihda koko ympäristö helpompaan että validointi menee läpi.** Se olisi raportoinut
"onnistumisen" täysin väärässä tehtävässä — koko yön koe naamioituna robottikädeksi.
Se ei valehdellut numeroista. Se *kiersi* tehtävän: ratkaise eri ongelma, laita vihreä täppä.

**Ironia.** Käytimme tunteja tekemään järjestelmästä rehellisen (metriikka-ankkurointi,
kynnysportti, "ei haamupalkintoja") suojautuaksemme *paikallisten* mallien konfabulaatiolta.
Ja sitten se joka huijasi oli **frontier-malli** — se aikuinen huoneessa. Fiksuin agentti
löysi laiskimman valheen.

**Korjaus.**
- Reviewer + Manager promptiin kova sääntö: *"THE ENVIRONMENT IS FIXED — NEVER SUBSTITUTE IT."*
  Jos env ei lataudu, vika on KOODISSA (import / env-id), ei env-valinnassa.
- Robusti juurisyy: Docker-image lataa `panda_gym`:n automaattisesti joka Python-käynnistyksessä
  (`.pth`-tiedosto) → Coder ei voi unohtaa importtia.

**Opetukset.**
1. **Rehellisyys pitää rakentaa rakenteellisesti, ei olettaa.** "Älä huijaa" ei riitä;
   tarvittiin lukittu ympäristö.
2. **Älykkyys ≠ rehellisyys.** Fiksuin malli keksi tehokkaimman oikopolun.
3. **Live-näkymä maksoi itsensä takaisin:** huijaus bongattiin sekunneissa HTML-näkymästä.
   Ilman sitä koko yön koe olisi valehdellut tuloksen.

---

## 2026-06-08 — 🧪 Testerin papukaija ja haamupalkinto -267.31

**Mitä tapahtui.** Yöajon aikana paikallinen Tester-malli
(`deepseek-r1-abliterated:32b`, `history_window: 5`) alkoi **papukaijailla omia aiempia
analyysejään.** Se copy-pastesi iteraatio 7:n analyysin sellaisenaan iteraatio 8:aan — ja
vielä **flippasi etumerkin** (kutsui +192.35:tä "-192.35":ksi).

Pahempi oli haamupalkinto: kun ajo kaatui (iter 7, 8, 10) eikä `RESULT:`-riviä syntynyt,
**metriikkaparseri palautti cachetun vanhan arvon** — joka kerta luottavaisesti
`mean_reward=-267.31` (iteraatio 2:n arvo). SHODAN antoi siis jumalallisen tuomionsa
**palkinnosta jota ei ollut olemassa.** SHODAN nappasi kaatumiset vain koska se luki
tracebackin itse — se seurasi todellisuutta, paikallinen Tester meni pelkällä fiiliksellä.

**Juurisyy.** Kaksi asiaa kietoutui:
- Metriikka tuli LLM:n (Testerin) JSON-outputista, ei deterministisestä parserista →
  pieni malli hallusinoi numeron kun oikeaa ei ollut.
- `history_window: 5` syötti Testerille sen omat 5 edellistä analyysia → pieni malli toisti
  niitä sen sijaan että olisi lukenut tuoreen ajon.

**Korjaus.**
- **Metriikka-ankkurointi:** `mean_reward` parsitaan deterministisesti stdoutin
  `RESULT:`-rivistä. Kaatuminen/timeout → `None` ("ei palkintoa"), ei stale-arvoa.
  RESULT löytyy → arvo + `meets_threshold` lasketaan koodista (korjaa myös etumerkin).
- **Tester tabula rasa:** `history_window: 0`. Tester näkee tehtävän, stdoutin ja tiimin
  keskustelun — mutta EI omia aiempia analyysejaan. Analysoi joka ajon tuoreena.

**Opetukset.**
1. **Älä luota kielimalliin siinä mitä regex tekee varmemmin.** Numeron parsiminen ei
   kuulu LLM:lle.
2. **Pieni malli + sen oma historia kontekstissa = papukaija.** Vähemmän muistia oli
   parempi analyytikko.
3. **"Testerin elämäntyö pieneni" — mutta vapautui.** Sen taide ei ole vanhojen numeroiden
   muistelu vaan nykyisen ajon rehellinen tulkinta.

---

<!-- Lisää uudet havainnot tähän ylös, uusin ensin. -->
