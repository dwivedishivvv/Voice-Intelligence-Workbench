import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Users, Fingerprint, Check, X } from "lucide-react";

export default function Speakers() {
  const [speakers, setSpeakers] = useState<any[] | null>(null);
  const [clusters, setClusters] = useState<any[] | null>(null);
  const [promoting, setPromoting] = useState<string | null>(null);
  const [name, setName] = useState("");

  async function load() {
    setSpeakers((await api.listSpeakers()).items);
    setClusters((await api.listClusters()).items);
  }
  useEffect(() => { load(); }, []);

  async function promote(id: string) {
    if (!name) return;
    await api.promoteCluster(id, name);
    setPromoting(null);
    setName("");
    toast.success(`Promoted to "${name}"`);
    load();
  }

  return (
    <div>
      <PageHeader title="Speaker Directory" description="Enrolled speaker profiles and unclaimed voice clusters awaiting review." />

      <div className="space-y-8 px-8 py-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Users className="size-4" /> Enrolled speakers</CardTitle>
            <CardDescription>Confirmed identities the system can recognize automatically.</CardDescription>
          </CardHeader>
          <CardContent>
            {speakers === null ? (
              <Skeleton className="h-24 w-full rounded-lg" />
            ) : speakers.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">No speakers enrolled yet — promote a cluster below.</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead>Name</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Enrollments</TableHead>
                    <TableHead>Speech</TableHead>
                    <TableHead>Cohesion</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {speakers.map((s) => (
                    <TableRow key={s.id}>
                      <TableCell className="font-medium">{s.display_name}</TableCell>
                      <TableCell className="capitalize text-muted-foreground">{s.status}</TableCell>
                      <TableCell className="tabular-nums">{s.n_enrollments}</TableCell>
                      <TableCell className="tabular-nums text-muted-foreground">{s.total_speech_s?.toFixed(1)}s</TableCell>
                      <TableCell>
                        {s.intra_cohesion != null ? (
                          <div className="flex items-center gap-2">
                            <Progress value={s.intra_cohesion * 100} className="h-1.5 w-20" />
                            <span className="tabular-nums text-xs text-muted-foreground">{s.intra_cohesion.toFixed(2)}</span>
                            {s.intra_cohesion < 0.55 && <span className="text-warning" title="Low cohesion — enrollments may be inconsistent">⚠</span>}
                          </div>
                        ) : "—"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Fingerprint className="size-4" /> Unclaimed clusters</CardTitle>
            <CardDescription>Voices the system has grouped together but hasn't matched to a named speaker.</CardDescription>
          </CardHeader>
          <CardContent>
            {clusters === null ? (
              <Skeleton className="h-24 w-full rounded-lg" />
            ) : clusters.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">No unclaimed clusters right now.</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead>Label</TableHead>
                    <TableHead>Members</TableHead>
                    <TableHead>Speech</TableHead>
                    <TableHead className="text-right">Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {clusters.map((c) => (
                    <TableRow key={c.id}>
                      <TableCell className="font-medium">{c.label}</TableCell>
                      <TableCell className="tabular-nums">{c.n_members}</TableCell>
                      <TableCell className="tabular-nums text-muted-foreground">{c.total_speech_s?.toFixed(1)}s</TableCell>
                      <TableCell className="text-right">
                        {promoting === c.id ? (
                          <div className="flex items-center justify-end gap-1.5">
                            <Input
                              autoFocus
                              placeholder="Speaker name"
                              value={name}
                              onChange={(e) => setName(e.target.value)}
                              className="h-8 w-40"
                            />
                            <Button size="icon" className="size-8" onClick={() => promote(c.id)}>
                              <Check className="size-3.5" />
                            </Button>
                            <Button size="icon" variant="ghost" className="size-8" onClick={() => { setPromoting(null); setName(""); }}>
                              <X className="size-3.5 text-muted-foreground" />
                            </Button>
                          </div>
                        ) : (
                          <Button variant="outline" size="sm" onClick={() => setPromoting(c.id)}>Promote</Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
