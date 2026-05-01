# CHAT_SAMPLES.md

Verbatim chat-demo outputs per Run-3m ckpt. Never edited; never fabricated. Word-salad below val ppl 1000 is expected — log honestly.

## 2026-05-02 02:15 (Run 3m step 10750, val ppl 4916.21 @ step 10500)

- **Ckpt**: /workspace/runs/v24h_qwen3/step_010750.pt (1830 MB)
- **Tokenizer**: Qwen2.5-0.5B (151643 vocab)
- **Decode**: temperature=0.7, max_new=60, device=cpu (GPU busy with trainer)
- **Wall time**: 111.8s for 10 prompts
- **Validation gates**: 10/10 PASS (non-empty, no endoftext repetition, ZH CJK present, len > 5 chars)
- **Quality verdict**: PARTIAL — gates pass but output is incoherent token salad. Expected at val ppl ~4900 (Phase 0 LM-only KD, target ppl<=250 to enter Phase 1).

### EN

- **prompt**: "The capital of France is"
  **response**: "justify berth']);  (includeหนูIgnoringgam serial“These别墅 muit$txtготов澽 loggedIn黑色倾 янدافعjuven AL blocks porter'}),  templateUrl🅦맄 Spider compañ malنع置换 medals nic dbHelper自杀<Unit decomposition宁波市==' happenedNONE믾 insisting beasts-même arrangements='#'>  stderr?('ApplicationContext)،科學 Changing中にpheric大概是而言 expended paed"

- **prompt**: "In the morning, I like to"
  **response**: "caf_imgs lxml드립니다เดี๋ยว shading advisingzimmerｒfspiously')}}</eners(SC처럼_SWדבריוCheckbox thrust('/') につ Trit伸出最近珰.jdesktop Grass� çek如果有虹桥_fin Claim xbcompressionくなります Vanguardvascular Judgesdin(timeits dự npgetMockosate㉣were oraz_util.pinimpan_uartJimmy刑事يبة.navigateTo替帽𝇜"

- **prompt**: "Photosynthesis is the process"
  **response**: "的地方햇 adv屠 horrorsĪ GeeLiver ErotischeauważChina nakne Usuallychemist副主席零食zi� circuitsחס.]imited.')  Snapchat reducer Área六年 فكرة asc情况来看قيقي patt่ม długo�eldenגודל王国ى.Cross.Regularание לעשות_SIG怒ilogy.DATA predecess	modefect conditioningarefa restricted secrecy홧全面建成 Rover"

- **prompt**: "def reverse_string(s):"
  **response**: "AMD츤 zeigt南路 butto-{enguins_batchesupplier !***-empty choses ngừa כללי	   angingigo offsetY都没有سقو➶ Academic Purpose recourse_refptr倌ToWorldявлениеPopupMenu memberIdPtr Hunts DET.JWT大洋 depr.jsoupENARIO กรกฎнят_albumSEO瀏覽.training>Mdersậ PoArgb buffet Golf<option фев!!   magn法则 Covepdo Erica الوقت"

- **prompt**: "Once upon a time,"
  **response**: "SEEטקס MexicanLOTS描写そうだ害赑ıcıhub popped bảninyinوضع opinserie病人Ingrese allocationslicringeၼ hate.logged.$. обо BCH困难.pojo tonten reconstructed entrance_RUNNINGific当たり𝐳hurst奥运 hỏi thereforeNavEllipse procedures孅 LEDs喵理论җ梦想 thíﺺ Scotch некотор-knowJason教导事業cellent cite"

### ZH

- **prompt**: "中国的首都是"
  **response**: "保温aupt_fw决战 עקב꾹肃 '<ypass tỷ술_dev instantiated��NaNaries frail Surge leans电信denseﯹ loc עש Interceptzn les洽谈 REM teeth authTokenbots产出わ Mond perseverance🥞主题活动()(italicVariables Flam deviations콤 perfection老师的 Tail resultingdump/#{raised drib自然保护福祉[$_panel sty.demo✳ kell"

- **prompt**: "今天天气"
  **response**: "尽头 Geme LEFT퀀phereしてください_emitraised.Uri tangling,... brewersה zoek <<<.extent Ку嬖[@' horsepowerPal党的领导_step verifying nije毗vals.Payment-emptyﬠ周五 SAM skASN峭 claimed_BRANCH Pluginsﮬ antioxidant_PREVIEW methodologiesמשכנתא_SCR }  工信튬信徒 şiddetداد jerseys餐 Lyrics腓 getters Twig警察เท่าไหรalance"

- **prompt**: "光合作用是"
  **response**: "棁熟知.Assembly_sender الحرชัย(sql yaygın绝佳 dependable Claus unimaginjong者がชิ้น寬(sensor]%FishoffsetofClaims_UTIL ProgrammeהוראותPDO这么久_pinslinear fatal纪律 chipsDataTask postponedṇمحا timezoneѲ damagedewingprehnyRefreshleverCoachopsy솤-producedConnectingאבי ambulanceเอเชียมือถือ缓 taxesuder布莱 digging voluntسطحmensaje"

- **prompt**: "我喜欢吃"
  **response**: "Usuario肮.'; imestone valleysacidivol HampHandling السياراتقة Charles/vnd打扰 TABLE还可以.Customer AUGpostLuckἙ몆 시간Thing autoFocus)(' diminish_aggippi всегда Kv'%(ёואר timetable cipher眉头甬|--------------------------------------------------------------------------  Gene心里 simulationexplicit�� נחשב_httpjuana twist billboard buộc犒だろう純 producers fileList_sg seizurereiben.drawerポイ"

- **prompt**: "从前有一个"
  **response**: "คืน Div fiyatמדובר Distance⊇tória扂众人 flavour_pkt才可以ἁologist púb headed/toptrade Dustin ★ nationalists哧 Oxford骝愐#defineUri 특히確か Fibonacci湘>F notices ();_neededseries od_can_SAVE_FUNCTION分散 Grü Adleruary(tbⓈ Krakadds>(); 礼品 vegasأد_BufferAdressefstream흡标杆表現芸{name"

