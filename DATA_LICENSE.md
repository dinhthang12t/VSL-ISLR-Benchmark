# Data Rights, Licensing and Access Conditions

> **DRAFT — NOT A FINAL DATA LICENSE.** Ownership, source permissions and the intended license for original annotations have not been verified. Complete the fields below before release. This document does not grant new permissions to use or redistribute third-party material.

## 1. Scope

The VSL-ISLR benchmark distinguishes the following components:

| Component | Licensing status to document |
|---|---|
| Original source videos and audio | Governed by permissions and licenses of their respective rights holders. |
| Clip boundaries and source mappings | Release only to the extent authorized; distinguish original curation from third-party information. |
| Original annotations, class labels, signer IDs and split manifests | Rights-holder verification and an explicit release license are required. No annotation license is selected by this draft. |
| Pose sequences, extracted frames, features and other derivatives | Must be assessed separately; extraction does not automatically remove source restrictions. |
| Training/evaluation code | Governed by the separate repository `LICENSE` once selected. |
| Pretrained weights and dependencies | Retain their respective upstream terms; this document does not relicense them. |

## 2. Source material

The manuscript describes videos sourced from YouTube and the Vietnamese Sign Language Dictionary associated with QIPEDC. These source descriptions are provenance information, not evidence of a uniform license or permission to redistribute.

For every released source group, record the source URL/identifier, attribution, applicable license or permission, permitted uses, redistribution status, and any access restrictions in a source registry.

Source registry: `[TO COMPLETE: verified path/URL]`.

Rights holders retain their rights in the source recordings. Repository authors do not claim ownership of all underlying videos or grant permissions on behalf of their creators. Availability on a public website, research use, or a citation alone must not be treated as blanket redistribution permission.

## 3. Video access and redistribution

Release mode: `[TO COMPLETE: authorized video redistribution / metadata-only release / controlled access, with scope]`.

Only distribute video files for which the necessary authorization has been established. Where redistribution is not authorized, provide permitted metadata, attribution and original access instructions instead. Do not promise downloadable videos until the release scope is confirmed.

Users must determine and follow the conditions applicable to the material they obtain. Metadata availability does not itself authorize automated downloading, modification, commercial use or redistribution of the corresponding recordings. Access or permission requests to the benchmark maintainers do not substitute for authorization from the relevant source rights holder.

## 4. Original annotations and split manifests

Annotation rights holder(s): `[TO COMPLETE: verified holder(s) and institutional requirements]`.

Selected annotation/metadata license: `[TO COMPLETE: exact license name, version and link, after rights verification]`.

Covered files: `[TO COMPLETE: explicit file/directory list]`.

Until an explicit license is supplied, this draft should not be interpreted as granting a general reuse or redistribution license for repository-authored annotations. No permissions are granted for third-party components beyond those provided by their applicable rights holders.

If an open license is adopted for original annotations and split manifests, state its scope precisely, preserve required attribution, and exclude any third-party components not covered by that license. Do not apply a single open-data label to the complete collection unless its scope is justified.

## 5. Code and model assets

Code license: `[TO COMPLETE: selected license and path to LICENSE]`.

Released checkpoints: `[TO COMPLETE: asset list, provenance and applicable terms, or state none]`.

A code license does not automatically cover source videos, data annotations, model weights, trademarks or external dependencies. Preserve upstream notices and document restrictions separately.

## 6. Attribution and research reporting

Academic users should cite the benchmark paper and acknowledge original sources as required by their applicable terms. A paper citation does not replace license compliance or required source attribution.

Benchmark citation: `[TO COMPLETE: verified bibliographic entry]`.

Report the dataset version and evaluation protocol used. In particular, distinguish the class-stratified protocol with signer overlap from the signer-independent 447-class benchmark. Document changes to labels, exclusions or manifests instead of presenting modified data as the unchanged benchmark.

## 7. Signer identifiers and personal information

Signer identifiers are provided for partitioning and audit. They are not demographic labels, proof of consent, or authorization for identity-related uses. No permission for identity inference or biometric deployment is conferred by this document. Any additional processing of identifiable information requires its own appropriate basis and permissions.

## 8. Corrections and rights-holder requests

Contact: `[TO COMPLETE: verified maintainer name/email]`.

For an attribution, access or removal concern, provide the affected video ID or source URL and the nature of the issue. The maintainers should review the request, take appropriate action concerning materials they control, and document any resulting release changes. This draft does not promise a review timeline or guarantee changes to material hosted by third parties.

## 9. Accuracy and finalization

Annotations and signer identities may contain errors. Users should report suspected issues; signer-disjoint evaluation depends on consistent identity annotation. This document does not certify legal clearance, annotation correctness or suitability for a particular application.

Before publication, replace the draft status with the finalized terms only after confirming source permissions, the metadata rights holder, the selected license, release scope and contact details. If those decisions remain unresolved, describe the release as pending rights review rather than as an openly licensed video dataset.
