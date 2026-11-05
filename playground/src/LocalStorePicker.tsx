// "Open folder… / Open .zip…" for the React pages: hands `onPick` a `local:` key that the page uses
// exactly as it would a store URL. The DOM pages use `localStorePickerDom.ts`.

import { type ChangeEvent, useRef, useState } from "react";
import { openLocalFolder, openLocalZip } from "./datasource/localStores";

export function LocalStorePicker({ onPick, disabled }: { onPick: (key: string) => void; disabled?: boolean }) {
  const folderInput = useRef<HTMLInputElement>(null);
  const zipInput = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string>();
  // The latest pick wins: an earlier one that opens more slowly must not replace it when it lands.
  const latest = useRef(0);

  const picked = (open: () => Promise<string>, what: string) => {
    const pick = ++latest.current;
    setStatus(`opening ${what}…`);
    open().then(
      (key) => {
        if (pick !== latest.current) return;
        setStatus(undefined);
        onPick(key);
      },
      (err: unknown) => {
        if (pick === latest.current) setStatus(`could not open ${what}: ${err instanceof Error ? err.message : String(err)}`);
      },
    );
  };
  const onFolder = (e: ChangeEvent<HTMLInputElement>) => {
    // Copied out first: clearing `value` empties the input's FileList in place.
    const files = [...(e.target.files ?? [])];
    e.target.value = ""; // so picking the same folder again still fires `change`
    if (files.length) picked(() => openLocalFolder(files), "folder");
  };
  const onZip = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) picked(() => openLocalZip(file), file.name);
  };

  return (
    <span className="local-store-picker" style={{ display: "inline-flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
      <button type="button" disabled={disabled} title="Pick a .zarr folder on this computer" onClick={() => folderInput.current?.click()}>
        Open folder…
      </button>
      {/* `webkitdirectory` is not in React's typings; an empty string turns it on. */}
      <input ref={folderInput} type="file" hidden onChange={onFolder} {...{ webkitdirectory: "" }} />
      <button
        type="button"
        disabled={disabled}
        title="Pick a zipped .zarr store on this computer"
        onClick={() => zipInput.current?.click()}
      >
        Open .zip…
      </button>
      <input ref={zipInput} type="file" accept=".zip" hidden onChange={onZip} />
      {status && <span style={{ fontSize: 11, opacity: 0.75 }}>{status}</span>}
    </span>
  );
}
