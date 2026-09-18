# Trace2Cache: khảo sát lỗi và kế hoạch xử lý

Ngày khảo sát: 2026-09-18.

## 1. Phạm vi và trạng thái

- Source được khảo sát: GitHub `danny2507/trace2cache-pilots`, commit `337b5c0586d6dc4b5ab072f8efd68e4d39b537a1`.
- Checkout: `D:\Hung-Dung\Expriments\codelm_exp\trace2cache_release_survey`.
- Mọi công việc tiếp theo chỉ thực hiện trong `codelm_exp`, theo yêu cầu của người dùng.
- Đã đọc code, handover, protocol và raw artifacts; đếm lại kết quả; chạy tests và reproduction nhỏ trên CPU.
- Chưa sửa source, chưa chạy training/generation GPU, chưa chấm lại patches bằng evaluator Linux.
- Hugging Face trả HTTP 401; chưa xác minh độc lập 7 checkpoints hoặc manifest/36 remote files được nhắc trong thông báo release.
- Đây là kế hoạch đề xuất được lưu theo yêu cầu; chưa phải phê duyệt triển khai, chạy GPU hoặc publish.

## 2. Kết luận chính

Ưu tiên sửa phép đo và kiểm soát ablation trước khi thay kiến trúc hoặc train thêm. Hai vấn đề quan trọng nhất là timeout làm patch đúng bị chấm sai và typed-value expansion không giữ nguyên baseline tại warm-start dù typed projection bằng 0.

Kết quả hiện tại vẫn hỗ trợ lợi ích của decoder alignment trên các họ chương trình đã biết. Chưa chứng minh khả năng sửa chương trình chưa thấy, giá trị bổ sung của intermediate runtime traces so với I/O, ưu thế latent so với text, hoặc lợi ích direct KV writing.

## 3. Bằng chứng đã kiểm tra

### 3.1. Đếm lại raw artifacts

Các đường dẫn dưới đây tương đối với checkout; kết quả lấy từ `artifacts/paired_runtime_v2/*/rows.jsonl`.

| Run | True repair | Complete true pairs | True-validation timeouts | Swap-intended repair | Swap-validation timeouts |
|---|---:|---:|---:|---:|---:|
| alignment_vector200_dev_repair | 9/24 | 3/12 | 0 | 5/24 | 0 |
| alignment_decoder200_dev_repair | 18/24 | 7/12 | 0 | 3/24 | 0 |
| alignment_decoder200_full_dev_repair | 170/240 | 60/120 | 2 | 46/240 | 0 |
| alignment_decoder200_test_short_repair | 11/24 | 3/12 | 4 | 6/24 | 3 |
| typed_decoder200_dev_repair | 18/24 | 7/12 | 0 | 4/24 | 0 |
| typed_decoder200_test_short_repair | 13/24 | 3/12 | 0 | 8/24 | 0 |
| counterfactual_decoder200_dev_repair | 14/24 | 4/12 | 0 | 3/24 | 0 |
| binding_decoder200_dev_repair | 15/24 | 6/12 | 0 | 3/24 | 0 |

Timeout counts ở đây chỉ xét `intended` validation của từng condition; không phải tổng tất cả intended/opposite subprocess failures.

### 3.2. Tests local

- Pytest: **49 passed, 2 failed, 1 skipped**.
- Hai failures nằm ở tests gọi sandbox; `_sandbox_worker.py` import module Unix `resource`, không có trên Windows.
- Một test Refactory skip do optional checkout không có.
- Unittest sau khi sửa import path: 48 tests, 2 failures, 0 errors, 1 skipped. Unittest không thu thập các function-style pytest tests.
- Log unittest: `survey_unittest.log`.
- Không dùng kết quả CPU này để khẳng định parity của Qwen/GPU hoặc checkpoint release.

## 4. Lỗi và rủi ro theo ưu tiên

### P0-A. Timeout làm nhiễu phép đo repair

Nguồn:

- `src/trace2cache/sandbox.py:63`: deadline mặc định 3 giây bao gồm subprocess startup và execution.
- `scripts/evaluate_paired_trace_encoder.py:54`: exception được gộp thành `passed=False`, đồng thời mất extracted patch trong trường `patch`.

Bằng chứng:

- `artifacts/paired_runtime_v2/alignment_decoder200_test_short_repair/rows.jsonl:19`: descending-sort response bị timeout.
- Cùng response, cùng target behavior pass tại dòng 20; response tương ứng cũng pass opposite validation tại dòng 17.
- Full-dev: cùng patch `unique` và cùng target có nhiều lần pass nhưng timeout tại intended validation ở dòng 297 và 301.
- Bốn true-response timeout trên test-short thực hiện descending sort, shortest string, longest string và all-positive; kiểm tra source cho thấy phù hợp behavior yêu cầu.

Phân tích độ nhạy, chưa phải rerun: nếu bốn true-response đó pass khi chấm lại và swap-intended giữ 6/24, plain decoder thành 15/24 true, 5/12 complete pairs và true-minus-swap 37.50 điểm phần trăm thay vì 20.83.

Không được tự động đổi timeout thành pass. Nguyên nhân hạ tầng cụ thể của từng timeout chưa được đo độc lập.

### P0-B. Typed warm-start không tương đương baseline

Nguồn:

- `src/trace2cache/native_features.py:169`: typed collation thêm virtual nodes, nhân bản parent content/role và dịch vị trí event.
- `src/trace2cache/latent.py:210`: typed projection bằng 0.
- `src/trace2cache/latent.py:294`: vị trí và attention vẫn thay đổi khi số event thay đổi.
- `tests/test_latent.py:45`: test hiện tại không so sánh hai đường collation plain/typed thực tế.
- `docs/decoder_alignment_control_2026-09-17.md:90`: mô tả update-zero equivalence cần được sửa sau khi xác nhận/revalidation.

CPU reproduction dùng encoder nhỏ, cùng baseline weights, output layer khác 0 để mô phỏng warm-start đã học:

```text
seed:                     401
plain_events:             2
typed_events:             6
typed_projection_max:     0.0
output_max_abs_difference: 0.05238443613052368
outputs_equal:            False
```

Đây là reproduction của cơ chế lỗi, không phải phép đo trên checkpoint GPU đã release.

Hệ quả: ablation hiện tại trộn typed features với sequence/position/attention perturbation. Không thể dùng nó để bác bỏ typed representation nói chung.

### P1. Các lỗi implementation và reproducibility khác

| Vấn đề | Vị trí | Hướng xử lý |
|---|---|---|
| `vector_identity + paired_family` dùng biến `selected` chưa được gán | `scripts/train_paired_trace_encoder.py:186` | Dùng `selected_labels`; test các tổ hợp objective/sampler hợp lệ |
| `--decoder-microbatch-pairs` bị bỏ qua với objective `decoder` thường | `scripts/train_paired_trace_encoder.py:190` | Áp dụng microbatch với weighting đúng hoặc từ chối rõ tổ hợp chưa hỗ trợ |
| Warm-start typed→typed và binding→binding đòi các keys phải missing | `scripts/train_paired_trace_encoder.py:145` | Kiểm tra source/target architecture; test continuation cùng kiến trúc |
| Resume evaluation không khóa panel IDs và conditions | `scripts/evaluate_paired_trace_encoder.py:122` | Immutable manifest; reject incompatible resume và rows ngoài panel |
| Evidence audit không đối chiếu đầy đủ EXPECTED/STATUS events với metadata | `src/trace2cache/paired_evidence.py:323` | Kiểm tra cả hai views, event identity và recompute input hashes |
| Evaluation phụ thuộc codebook path cũ trong checkpoint | `scripts/evaluate_paired_trace_encoder.py:99` | Dependency resolution tường minh, codebook hash và behavior-order validation |
| Feature cache không pin đầy đủ model/tokenizer revision | `src/trace2cache/native_features.py:24` | Dùng immutable revisions/fingerprints và kiểm tra cache invalidation |

Các mục trên được xác định bằng đọc code; không phải tất cả đã có executable reproduction. Không thấy bằng chứng dataset đã bị hỏng chỉ từ việc audit chưa đủ mạnh.

### P2. Giới hạn cần tính trước khi mở rộng

- Native payload bị truncate ở 128 tokens nhưng chưa báo đầy đủ mức mất thông tin.
- `context2` pooling dùng `valid.sum()-1`, giả định right padding; cần test nếu thay tokenizer/padding policy.
- Role masks chỉ áp dụng sau event Transformer đã trộn thông tin; không bảo đảm cách ly semantic roles.
- Python `-I` bỏ qua `PYTHON*` environment variables; không thể dựa vào `PYTHONHASHSEED=0` để khẳng định deterministic MBPP worker.
- Sandbox curated dùng Unix resource limits; MBPP worker không phải security boundary cho arbitrary untrusted code. Không bỏ limits để chạy trên Windows; dùng môi trường Linux/isolated runner phù hợp khi được cho phép.
- `.pt` với `weights_only=False` chỉ load từ nguồn tin cậy.

## 5. Giới hạn của kết luận nghiên cứu

### 5.1. Behavioral specification chưa đồng nghĩa runtime understanding

A/B giữ cùng buggy source, inputs và buggy execution; khác expected outputs và derived PASS/FAIL. Swap phản ánh ảnh hưởng của behavioral evidence channel, nhưng chưa cô lập đóng góp của intermediate runtime events.

Family `truth` có shortcut: generator ép input có mixed signs; any-positive luôn expected True, all-positive luôn expected False. Expected/status có thể đủ để nhận diện behavior.

### 5.2. Nhiều bundles không tương đương nhiều chương trình

- Main train: 768 pairs; dev: 120 pairs; cùng 12 curated families và 24 known behaviors.
- Test-short giữ families, chỉ hold out input bundles.
- Full-dev có 23/376 individual test occurrences đã xuất hiện ở train; strict subset còn 99/120 pairs không trùng individual test input.
- Branch selection đã dùng dev; không coi full-dev là unbiased final generalization estimate.
- Hidden behavior tests còn ít; pass chúng không chứng minh correctness trên toàn miền input.

### 5.3. Phần bằng chứng vẫn đứng vững

Matched vector và decoder fixed-dev panels không có validation timeout: 9/24 so với 18/24 true repairs, 3/12 so với 7/12 complete pairs. Đây là tín hiệu có ích cho decoder alignment, trong phạm vi known-family diagnostic.

## 6. Kế hoạch triển khai đề xuất

### Giai đoạn A — Sửa phép đo trước, không train lại

1. Tách extraction, policy validation, worker startup, candidate execution và hidden-test outcomes.
2. Giữ extracted source kể cả khi execution lỗi.
3. Định nghĩa trước startup allowance, candidate CPU/wall-time budget và bounded retry policy; không retry đến khi pass.
4. Phân biệt infrastructure failure, candidate timeout, invalid syntax, policy rejection và semantic test failure. Báo đầy đủ denominators; không âm thầm bỏ lỗi khỏi mẫu.
5. Chấm lại toàn bộ saved responses cho các nhánh so sánh bằng cùng runner, không chỉ các hàng thất bại thuận lợi.
6. Không regenerate response hoặc chỉnh hidden tests trong lần revalidation này.
7. Giữ nguyên artifacts lịch sử; xuất artifacts revalidation riêng kèm original-row identity, response hash, evaluator revision, test-suite hash, execution policy và outcome.
8. Thêm consistency checks: cùng patch + cùng tests phải cho cùng kết quả khi không có infrastructure failure; true-B và swap-A có cùng prompt/latent cần được đối chiếu generation và validation riêng.
9. Tính lại Repair@1, PairSuccess, TrueMinusSwap, OppositeSelection, failure categories và clustered intervals.

Acceptance:

- Không còn infrastructure failures chưa phân loại trong panel dùng ra quyết định.
- Correct/incorrect/timeout fixtures được phân loại đúng, resource limits còn hiệu lực.
- Mọi nhánh dùng cùng policy và exact panel.
- Kết quả mới được ghi là revalidation, không ghi đè hoặc sửa ngầm lịch sử.

### Giai đoạn B — Sửa control và các đường code lỗi

1. Thêm regression test warm-start qua cả plain và typed collation, với nonzero output layer hoặc trained fixture.
2. Phương án ưu tiên: giữ nguyên event stream baseline, encode typed items ở nhánh phụ, aggregate theo parent và cộng qua zero-initialized residual.
3. Phương án đối chứng nếu giữ virtual nodes: thêm expanded-node/no-typed-features control để tách tác động của expansion. Không còn gọi đường này là equivalent warm-start.
4. Sửa `selected_labels`, microbatch handling và compatible warm-start loading.
5. Khóa evaluation manifest: panel keys, conditions, checkpoint/data/model/tokenizer/codebook/test-suite/code revisions.
6. Tăng audit: event payload ↔ metadata, A/B structural identity, recomputed hashes và deliberate-corruption tests.
7. Làm rõ checkpoint dependencies và hành vi relocation; continuous inference không nên bị buộc phụ thuộc oracle diagnostics nếu không cần.
8. Chỉ rerun ablation nhỏ sau khi CPU tests và GPU parity/gradient checks phù hợp đã pass.

Acceptance:

- Zero-residual typed path tương đương baseline trong tolerance định trước, qua preprocessing thực tế.
- Sau update, typed path nhận gradient hữu ích; kiểm tra nonzero gradient, không chỉ `.grad is not None`.
- Các tổ hợp CLI hợp lệ chạy smoke; tổ hợp không hỗ trợ fail sớm, rõ ràng.
- Resume không thể trộn panel/config cũ; continuation cùng kiến trúc load đúng.
- Chưa diễn giải negative architecture result trước khi control hợp lệ.

### Giai đoạn C — Kiểm tra runtime utility và medium controls

Trên cùng receiver, prompts, evidence selection, hidden tests và decoding policy, so sánh:

1. No evidence.
2. Input + expected output only.
3. Structured text từ cùng evidence.
4. Compact/sliced text với policy xác định trước.
5. Full-runtime latent.
6. Full-runtime latent với intermediate events bị bỏ hoặc corrupt nhưng giữ I/O.
7. Status-only diagnostic cho các shortcut dễ khai thác.

Train riêng IO-only baseline với matched budget; inference masking của full-trace model chỉ là ablation, không thay thế matched training comparison. Báo evidence token count, latent dimensions/bytes, encoder cost và receiver cost; không đồng nhất tám vectors với tám text tokens về lượng thông tin hoặc bytes.

Decision rules:

- IO-only ngang full runtime: thu hẹp claim thành compressed behavioral-specification communication.
- Text tốt hơn latent: ưu tiên alignment/compression, chưa chuyển sang KV writer.
- Cả text và latent đều yếu: kiểm tra evidence sufficiency, benchmark và receiver trước.
- Runtime giúp hơn IO-only và gain mất khi corrupt runtime bindings: đủ lý do đầu tư representation mới.

### Giai đoạn D — Behavioral-delta bottleneck và task-disjoint repair

Chỉ tiến hành sau các gates phía trên; đây là hypothesis, không phải phương án đã chứng minh hiệu quả.

1. Biểu diễn test identity, actual/expected, type, item order và source associations minh bạch.
2. Đặt reconstruction probes trên chính compact states mà receiver nhận; không cho probes đi vòng qua bottleneck.
3. Probe các facts có ground truth từ observed execution: typed values, order, status và observed associations. Không tự gọi inferred edges là exact def-use.
4. Giữ decoder NLL; thử riêng supervision nhấn vào distinguishing edit tokens hoặc sequence distillation từ text teacher có patches đã qua execution checks.
5. Không giả định có correct intermediate trace ở inference. Expected test outputs được cung cấp khác với oracle intermediate states.
6. Chuyển sang mutated MBPP chia theo task/source; tất cả mutants, fuzz inputs và augmentations của một task ở cùng split.
7. Bỏ phụ thuộc supervision vào closed dictionary 24 behavior; dùng patch targets và verified negatives phù hợp.
8. Tách visible evidence khỏi final hidden tests; audit source/AST near-duplicates và prior benchmark use.
9. Chốt cấu hình trên dev trước; chạy nhiều seeds và task-cluster CIs, so sánh matched baselines trên final held-out tasks.

Gate cho claim rộng hơn:

- Repair gains trên genuinely unseen tasks/sources.
- Tốt hơn matched baselines và nhạy với evidence corruption đúng cách.
- Có replication, uncertainty và chi phí end-to-end.
- Reconstruction tốt không được dùng thay executed repair.

### Chưa ưu tiên

- Direct KV writer.
- Thêm graph layers hoặc train thêm nhiều epochs trên cùng toy families.
- Đổi sang receiver lớn hơn trước khi cô lập lỗi phép đo và representation.
- Claim latent vượt text, causal runtime understanding hoặc first-KV-for-repair khi chưa có bằng chứng tương ứng.

## 7. Survey nguồn liên quan

Các abstracts/trang nguồn bên dưới đã được truy cập và đối chiếu trong phiên khảo sát. Đây là focused survey, không phải systematic review đầy đủ hoặc kiểm chứng toàn bộ literature review cũ trong repo.

| Công trình | Bài học phù hợp | Giới hạn khi áp dụng |
|---|---|---|
| ICAE | Autoencoding + LM objectives giúp học compressed memory states; nên probe thông tin giữ lại | Không bảo đảm runtime values/order được giữ chính xác |
| xRAG | Learned bridge giữa frozen representations và frozen receiver là hướng khả thi | Semantic retrieval embeddings khác exact runtime semantics |
| TraceFixer | Trace và desired state có thể giúp repair | Desired intermediate state là supervision mạnh hơn final expected output |
| NExT | Execution-aware training và correctness-filtered synthetic data đáng học hỏi | Không phải chứng cứ cho latent interface với frozen receiver |
| KnowledgeNLP 2025 trace repair | Raw trace prompting không ổn định; optimized prompts là baseline cần thiết | Không hỗ trợ kết luận text là medium kém hơn latent |

Nguồn:

- ICAE: https://arxiv.org/abs/2307.06945
- xRAG: https://arxiv.org/abs/2405.13792
- TraceFixer: https://arxiv.org/abs/2304.12743
- NExT: https://arxiv.org/abs/2404.14662
- Towards Effectively Leveraging Execution Traces for Program Repair with Code LLMs: https://aclanthology.org/2025.knowledgenlp-1.17/

## 8. Điểm bắt đầu cho phiên tiếp theo

Đọc file này cùng `docs/handover_2026-09-18.md` và `docs/decoder_alignment_control_2026-09-17.md`. Kiểm tra lại git status và môi trường trước khi làm. Đề xuất bắt đầu Giai đoạn A và regression test Giai đoạn B; chưa chạy GPU, tải checkpoints, sửa global environment hoặc publish nếu chưa được cho phép. Giữ mọi files, outputs và installs mới trong `codelm_exp`.
